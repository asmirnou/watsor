import threading
import json
from os import getpid, getcwd, path
from platform import node
from pathlib import Path
from functools import partial
from signal import signal, SIGINT
from subprocess import PIPE, DEVNULL
from textwrap import dedent
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue, Event, Process, BoundedSemaphore, set_start_method
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from yaml.parser import ParserError
from werkzeug.serving import make_server
from werkzeug.routing import Map
from werkzeug.routing import Rule
from werkzeug.wrappers import Request, Response
from werkzeug.exceptions import HTTPException, BadRequest
from watsor.config.loader import parse, validate, normalize
from watsor.stream.sync import BalancedQueue, CountableQueue
from watsor.stream.log import LogHandler
from watsor.stream.watch import WatchDog
from watsor.stream.share import FrameBuffer
from watsor.stream.ffmpeg import FFmpegDecoder, FFmpegEncoder, MpegTSReader
from watsor.detection.detector import create_object_detectors
from watsor.output.video import VisualEffects, MotionJpeg, MpegTS
from watsor.filter.confidence import ConfidenceFilter
from watsor.filter.area import AreaFilter
from watsor.filter.mask import MaskFilter
from watsor.filter.track import TrackFilter
from watsor.filter.sieve import DetectionSieve
from watsor.output.draw import DrawEffect, DrawEffectWithContours
from watsor.output.blend import BlendEffect
from watsor.output.copy import CopyImageEffect, CopyHeaderEffect
from watsor.output.mqtt import MQTT
from watsor.output.snapshot import Snapshot

Camera = namedtuple('Camera',
                    ['frame_buffer_in', 'frame_buffer_out',
                     'decoder', 'encoder', 'sieve', 'mqtt', 'snapshot',
                     'visual_effects', 'visual_effects_queue', 'jpeg_encoder_buffer',
                     'mpegts_reader', 'mpegts_buffer'])


class _BasicApp:

    @property
    def app_name(self):
        return Path(__file__).parent.stem

    def _parse_commandline_arguments(self):
        parser = ArgumentParser(description='Object detection for video surveillance')
        parser.add_argument('-c', "--config",
                            dest='config_file_name', metavar='CONFIG_FILE_NAME',
                            required=True, help='configuration file')
        parser.add_argument("--model-path",
                            dest='model_path', metavar='MODEL_PATH',
                            default=path.join(getcwd(), 'model'),
                            help="path to log file")
        parser.add_argument("--log-path",
                            dest='log_path', metavar='LOG_PATH',
                            default=getcwd(),
                            help="path to log file")
        parser.add_argument('--log-level',
                            dest='log_level', metavar='LOG_LEVEL',
                            type=str, choices=['debug', 'info', 'warning', 'error', 'fatal'],
                            default='info', help='log level')

        self._args = parser.parse_args()
        self._args.log_level = self._args.log_level.upper()

    def _install_signal_handler(self):
        self._stop_main_event = threading.Event()
        signal(SIGINT, partial(lambda stop_event, *_args: stop_event.set(), self._stop_main_event))

    def _init_logging(self):
        self._stop_logging_event = threading.Event()
        self._log_queue = CountableQueue()

        self._logger = getLogger()
        self._logger.addHandler(QueueHandler(self._log_queue))
        self._logger.setLevel(self._args.log_level)

        filename = path.join(self._args.log_path, '{}.log'.format(self.app_name))

        self._log_handler = LogHandler(threading.Thread, "logger", self._stop_logging_event, self._log_queue,
                                       filename=filename, kwargs={'log_level': self._args.log_level})
        self._log_handler.start()

    def _stop_logging(self):
        self._log_queue.join()
        self._stop_logging_event.set()
        self._log_handler.join(30)

    def _read_config(self):
        self._config = normalize(validate(parse(self._args.config_file_name)),
                                 path.dirname(self._args.config_file_name))

    def _init_watch_dog(self):
        self._stop_watch_dog_event = threading.Event()
        self._watch_dog = WatchDog("watchdog", self._stop_watch_dog_event, self._log_queue,
                                   kwargs={'log_level': self._args.log_level})
        self._watch_dog.add_child(self._log_handler)
        self._watch_dog.start()

    def _stop_watch_dog(self):
        self._stop_watch_dog_event.set()
        self._watch_dog.join(30)


class _HTTPApplication(_BasicApp):

    def __init__(self):
        self._cameras = {}
        self._stop_events = []
        self._detectors = []

    def _http_serve(self):
        rules = [Rule("/", methods=["GET"], endpoint="home"),
                 Rule("/health", methods=["GET"], endpoint="health"),
                 Rule("/metrics", methods=["GET"], endpoint="metrics"), ]
        for camera in self._config['cameras']:
            camera_name = next(iter(camera))

            rules.append(Rule("/snapshot/{}/<label>".format(camera_name),
                              defaults={'camera_name': camera_name},
                              methods=["GET"], endpoint="snapshot"))
            rules.append(Rule("/video/mjpeg/{}".format(camera_name),
                              defaults={'camera_name': camera_name},
                              methods=["GET"], endpoint="stream_video_mjpeg"))
            if self._cameras[camera_name].mpegts_reader is not None:
                rules.append(Rule("/video/mpegts/{}".format(camera_name),
                                  defaults={'camera_name': camera_name},
                                  methods=["GET"], endpoint="stream_video_mpegts"))
        self._url_map = Map(rules)

        self._server = make_server('', self._config['http']['port'], self._dispatch_request, threaded=True)

        log = getLogger('werkzeug')
        log.setLevel(self._args.log_level)
        log.info("Listening on {}".format(self._server.socket.getsockname()))

        self._server_thread = threading.Thread(target=self._server.serve_forever)
        self._server_thread.start()

    def _stop_http(self):
        self._server.shutdown()
        self._server_thread.join(30)

    def _dispatch_request(self, environ, start_response):
        request = Request(environ)
        try:
            if self._check_auth(request.authorization):
                endpoint, values = self._url_map.bind_to_environ(environ).match()
                response = getattr(self, f"_on_{endpoint}")(request, **values)
            else:
                response = self._auth_required(request)
        except HTTPException as e:
            response = e
        return response(environ, start_response)

    def _check_auth(self, auth):
        return 'username' not in self._config['http'] or \
               (auth and
                auth.username == self._config['http']['username'] and
                ('password' not in self._config['http'] or
                 auth.password == self._config['http']['password']))

    def _auth_required(self, request):
        return Response("You have to login with proper credentials.", 401,
                        {"WWW-Authenticate": 'Basic realm="Access to {}"'.format(self.app_name.capitalize())})

    def _on_home(self, request):
        response = Response(mimetype="text/html")
        response.stream.write(dedent("""\
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>{title}</title>
            </head>
            <body>
            <dl>
                <dt><p>Cameras:</p></dt>
                {cameras}
            </dl>
            <p><a href="/metrics">Metrics</a></p>
            <p><a href="/health">Health</a></p>        
            </body>
            </html>
            """).format(
            title=self.app_name.capitalize(),
            cameras="\n\t".join(map(
                lambda camera, camera_name:
                "<dd><p><em>{name}</em>: "
                "video in <a href=\"/video/mjpeg/{name}\">Motion JPEG</a>, "
                "{mpegts}"
                "snapshot of {snapshots}"
                "</p></dd>".format(
                    name=camera_name,
                    mpegts="video in <a href=\"/video/mpegts/{name}\">MPEG-TS</a>, ".format(name=camera_name)
                    if self._cameras[camera_name].mpegts_reader is not None else "",
                    snapshots=", ".join([
                        "<a href=\"/snapshot/{name}/{label}\">{label}</a>".format(
                            name=camera_name,
                            label=next(iter(entry)))
                        for entry in camera[camera_name]['detect']
                    ])),
                self._config['cameras'],
                map(lambda camera: next(iter(camera)), self._config['cameras']))
            )))
        return response

    @staticmethod
    def _on_health(request):
        return Response('UP', mimetype='text/plain')

    def _on_metrics(self, request):
        metrics = defaultdict(list)

        for camera_name, entry in self._cameras.items():
            camera = {
                'name': camera_name,
                'fps': {
                    'decoder': round(entry.decoder.fps(), 1),
                    'sieve': round(entry.sieve.fps(), 1),
                    'visual_effects': round(entry.visual_effects.fps(), 1),
                    'snapshot': round(entry.snapshot.fps(), 1)
                },
                'buffer_in': round(entry.frame_buffer_in.fullness * 100),
                'buffer_out': round(entry.frame_buffer_out.fullness * 100)
            }
            if entry.encoder is not None:
                camera['fps']['encoder'] = round(entry.encoder.fps(), 1)
            if entry.mqtt is not None:
                camera['fps']['mqtt'] = round(entry.mqtt.fps(), 1)
            metrics['cameras'].append(camera)

        for detector in self._detectors:
            inference_time = detector.inference_time()
            max_fps = round(1000 / inference_time) if inference_time > 0 else 0.0

            metrics['detectors'].append({
                'name': str(detector.device_name, 'utf-8'),
                'fps': round(detector.fps(), 1),
                'fps_max': max_fps,
                'inference_time': round(inference_time, 1)
            })

        return Response(json.dumps(metrics, indent=4), mimetype='application/json')

    def _on_snapshot(self, request, camera_name, label):
        try:
            _, jpg = self._cameras[camera_name].snapshot.get(label)
            return Response(jpg.tobytes(), content_type="image/jpeg")
        except AssertionError as e:
            return BadRequest(e)

    def _on_stream_video_mjpeg(self, request, camera_name):
        encoder_queue = Queue(1)
        subscriptions = {
            self._cameras[camera_name].sieve: self._cameras[camera_name].visual_effects_queue,
            self._cameras[camera_name].visual_effects: encoder_queue
        }
        encoder = MotionJpeg(camera_name, self._stop_events[0], self._log_queue, encoder_queue,
                             self._cameras[camera_name].frame_buffer_out,
                             self._cameras[camera_name].jpeg_encoder_buffer, subscriptions,
                             kwargs={'log_level': self._args.log_level})

        response = Response(encoder, mimetype=encoder.mime_type)
        response.call_on_close(encoder.close)
        return response

    def _on_stream_video_mpegts(self, request, camera_name):
        encoder_queue = Queue(1)
        subscriptions = {
            self._cameras[camera_name].mpegts_reader: encoder_queue
        }
        encoder = MpegTS(camera_name, self._stop_events[0], self._log_queue, encoder_queue,
                         self._cameras[camera_name].mpegts_buffer, subscriptions,
                         kwargs={'log_level': self._args.log_level})

        response = Response(encoder, mimetype=encoder.mime_type)
        response.call_on_close(encoder.close)
        return response


class Application(_HTTPApplication):

    @staticmethod
    def _create_filters(camera_config):
        filters = [ConfidenceFilter(camera_config),
                   AreaFilter(camera_config)]
        if 'mask' in camera_config:
            filters.append(MaskFilter(camera_config))
        return [TrackFilter(filters)]

    @staticmethod
    def _create_effects(camera_config):
        # noinspection PyListCreation
        effects = []
        effects.append(CopyHeaderEffect())
        if 'mask' in camera_config:
            effects.append(BlendEffect(camera_config))
            effects.append(DrawEffectWithContours(camera_config))
        else:
            effects.append(CopyImageEffect())
            effects.append(DrawEffect())
        return effects

    def _create_encoder(self, camera_config, camera_name, frame_buffer_out, buffer_size,
                        detection_sieve, visual_effects, visual_effects_queue):
        if 'encoder' not in camera_config['ffmpeg']:
            return None, None, None

        encoder_queue = Queue(1)
        encoder = FFmpegEncoder(camera_name, self._stop_events[0], self._log_queue, encoder_queue,
                                frame_buffer_out, camera_config['ffmpeg']['encoder'],
                                DEVNULL if 'output' in camera_config else PIPE,
                                kwargs={'log_level': self._args.log_level})
        self._processes.append(encoder)
        detection_sieve.subscribe(visual_effects_queue)
        visual_effects.subscribe(encoder_queue)

        if 'output' in camera_config:
            return encoder, None, None

        mpegts_buffer = FrameBuffer(buffer_size, int(camera_config['width'] / 4), 188, 1)

        mpegts_reader = MpegTSReader(camera_name, self._stop_events[0], self._log_queue,
                                     encoder.stdout, mpegts_buffer,
                                     kwargs={'log_level': self._args.log_level})
        self._processes.append(mpegts_reader)

        return encoder, mpegts_reader, mpegts_buffer

    def _create_mqtt(self, camera_config, camera_name, frame_buffer_in, decoder, decoder_stop_event,
                     detection_sieve):
        if 'mqtt' not in self._config:
            return None

        mqtt_queue = Queue(1)
        mqtt = MQTT(Process, camera_name, self._stop_events[0], self._log_queue, mqtt_queue,
                    frame_buffer_in, decoder.fps, decoder.rate_limiter, decoder_stop_event,
                    self._config['mqtt'], camera_config,
                    kwargs={'topic': self.app_name, 'log_level': self._args.log_level})
        self._processes.append(mqtt)
        detection_sieve.subscribe(mqtt_queue)
        return mqtt

    def _setup(self):
        self._processes = []
        self._stop_events += [Event()]
        self._frame_queue = Queue()

        all_semaphores = {}
        for camera in self._config['cameras']:
            camera_name = next(iter(camera))
            camera_config = camera[camera_name]

            buffer_size = 10
            frame_buffer_in = FrameBuffer(buffer_size, camera_config['width'], camera_config['height'])
            frame_buffer_out = FrameBuffer(buffer_size, camera_config['width'], camera_config['height'])

            decoder_stop_event = Event()
            decoder_queue_semaphore = BoundedSemaphore(1)
            all_semaphores[camera_name] = decoder_queue_semaphore
            decoder_queue = BalancedQueue(self._frame_queue, {camera_name: decoder_queue_semaphore}, camera_name)
            decoder = FFmpegDecoder(camera_name, decoder_stop_event, self._log_queue, decoder_queue,
                                    frame_buffer_in, camera_config['ffmpeg']['decoder'],
                                    kwargs={'log_level': self._args.log_level})
            self._processes.append(decoder)
            self._stop_events.append(decoder_stop_event)

            filters = self._create_filters(camera_config)
            detection_sieve_queue = Queue(1)
            detection_sieve = DetectionSieve(camera_name, self._stop_events[0], self._log_queue,
                                             detection_sieve_queue, frame_buffer_in, filters, decoder.rate_limiter,
                                             kwargs={'log_level': self._args.log_level})
            self._processes.append(detection_sieve)
            decoder.subscribe(detection_sieve_queue)

            visual_effects_queue = Queue(1)
            visual_effects = VisualEffects(camera_name, self._stop_events[0], self._log_queue,
                                           visual_effects_queue, frame_buffer_in, frame_buffer_out,
                                           self._create_effects(camera_config),
                                           kwargs={'log_level': self._args.log_level})
            self._processes.append(visual_effects)

            encoder, mpegts_reader, mpegts_buffer \
                = self._create_encoder(camera_config, camera_name, frame_buffer_out, buffer_size,
                                       detection_sieve, visual_effects, visual_effects_queue)

            mqtt = self._create_mqtt(camera_config, camera_name, frame_buffer_in, decoder,
                                     decoder_stop_event, detection_sieve)

            snapshot_queue = Queue(1)
            snapshot = Snapshot(camera_name, self._stop_events[0], self._log_queue, snapshot_queue,
                                frame_buffer_in, camera_config,
                                self._create_effects(camera_config),
                                kwargs={'topic': self.app_name, 'log_level': self._args.log_level})
            self._processes.append(snapshot)
            detection_sieve.subscribe(snapshot_queue)

            self._cameras[camera_name] = Camera(frame_buffer_in, frame_buffer_out,
                                                decoder, encoder, detection_sieve, mqtt, snapshot,
                                                visual_effects, visual_effects_queue,
                                                MotionJpeg.create_buffer(buffer_size),
                                                mpegts_reader, mpegts_buffer)

        self._detectors += create_object_detectors(Process, self._stop_events[0], self._log_queue,
                                                   BalancedQueue(self._frame_queue, all_semaphores),
                                                   {n: c.frame_buffer_in for n, c in self._cameras.items()},
                                                   self._args.model_path,
                                                   kwargs={'log_level': self._args.log_level})
        self._processes += self._detectors

    def _start(self):
        self._logger.info("Starting {} on {} with PID {}".format(
            self.app_name.capitalize(), node(), getpid()))

        for process in self._processes:
            process.start()
            self._watch_dog.add_child(process)

    def _stop(self):
        self._logger.info("Stopping {}".format(self.app_name.capitalize()))

        for stop_event in self._stop_events:
            stop_event.set()

        for process in self._processes:
            process.join(30)

    def _terminate(self):
        for process in self._processes:
            process.terminate()

    def run(self):
        self._parse_commandline_arguments()
        self._install_signal_handler()
        self._init_logging()
        try:
            self._read_config()
            self._init_watch_dog()
            try:
                self._setup()
                self._http_serve()
                try:
                    self._start()
                    self._stop_main_event.wait()
                    self._stop()
                finally:
                    self._stop_http()
            except Exception:
                self._terminate()
                raise
            finally:
                self._stop_watch_dog()
        except (ValueError, AssertionError, ParserError, FileNotFoundError, OSError) as e:
            self._logger.error(e)
            exit(1)
        except Exception as e:
            self._logger.exception(e)
            exit(1)
        finally:
            self._stop_logging()


if __name__ == '__main__':
    set_start_method('spawn')

    app = Application()
    app.run()
