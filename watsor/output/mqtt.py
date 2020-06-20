import json
import re
import paho.mqtt.client as mqtt
from distutils.util import strtobool
from time import time
from datetime import datetime
from threading import current_thread, RLock
from collections import namedtuple, defaultdict
from watsor.stream.work import Work, WorkPublish, Payload
from watsor.stream.share import FrameBuffer, FramesPerSecond
from watsor.config.coco import get_coco_class

UserData = namedtuple('userdata', ['topic', 'rate_limiter', 'stop_event'])


class ReportedState(object):

    def __init__(self):
        self.state = False
        self.when = 0.0


class NamedClient(mqtt.Client):

    def _thread_main(self):
        current_thread().name = self._client_id.decode('utf-8')
        super()._thread_main()

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port


# noinspection PyUnusedLocal
class MQTT(WorkPublish):
    """Communicates with HomeAssistant and others via MQTT.
    Reports state of the camera and its sensors such as FPS.
    Publishes binary state of a detected object, confirming the state every 10 seconds.
    Allows to start/stop the decoder, limit frame rate and enable/disable the reporting of detection details.

    List of topics:
    - watsor/cameras/{camera}/available
    - watsor/cameras/{camera}/command
    - watsor/cameras/{camera}/sensor
    - watsor/cameras/{camera}/state
    - watsor/cameras/{camera}/detection/{label}/state
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer,
                 decoder_fps, decoder_rate_limiter, decoder_stop_event, mqtt_config, camera_config, kwargs=None):
        self.__fps = FramesPerSecond()
        self.__states = self.__init_states(camera_config)
        self.__sensors_hash = 0
        self.__old_state = None
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue, frame_buffer,
                         args=(mqtt_config, self.__fps, decoder_fps, decoder_rate_limiter, decoder_stop_event),
                         kwargs={} if kwargs is None else kwargs)

    @staticmethod
    def __init_states(camera_config):
        """Initialize the dictionary of detection states in order to start reporting OFF
        even if nothing's detected.

        :param camera_config: detection configuration for a camera
        :return: initialized default dictionary
        """
        result = defaultdict(ReportedState)
        for entry in camera_config['detect']:
            coco_class = next(iter(entry))
            _ = result[coco_class]
        return result

    @property
    def fps(self):
        return self.__fps

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Work, self)._run(stop_event, log_queue, *args, **kwargs)
        client, user_data = self._init_client(*args, **kwargs)
        if client is None:
            return
        try:
            self._init_locals()
            self._spin(self._process, stop_event, *args[:3], *args[4:], client, user_data, **kwargs)
        except Exception:
            self._logger.exception('Spin failure')
        finally:
            self._close_client(client, user_data)

    def _init_locals(self):
        self.__details = False
        self.__command_lock = RLock()
        self.__command_fps = re.compile(r"^fps\s*=\s*(\d+)$", re.IGNORECASE)
        self.__command_details = b = re.compile(r"^details\s*=\s*(\w+)$", re.IGNORECASE)

    def _init_client(self, frame_queue, stop_event, frame_buffer, config, fps, decoder_fps,
                     decoder_rate_limiter, decoder_stop_event, *args, **kwargs):
        try:
            client = NamedClient(client_id=self.name)

            if 'username' in config:
                client.username_pw_set(config['username'], config['password'])

            user_data = UserData(topic="{}/cameras/{}".format(kwargs['topic'], self.name),
                                 rate_limiter=decoder_rate_limiter,
                                 stop_event=decoder_stop_event)
            client.user_data_set(user_data)

            client.will_set(user_data.topic + '/available', payload='offline', qos=1, retain=True)

            client.on_log = self._on_log
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message

            client.connect(config['host'], config['port'])

            client.loop_start()
            return client, user_data
        except Exception as e:
            self._logger.error(e)
            return None, None

    def _close_client(self, client, user_data):
        try:
            client.loop_stop()
        except Exception:
            self._logger.exception('MQTT client failure')

    def _on_log(self, client, userdata, level, buf):
        if level < mqtt.MQTT_LOG_DEBUG:
            level_std = mqtt.LOGGING_LEVEL[level]
            self._logger.log(level_std, buf)

    def _on_connect(self, client, user_data, flags, rc):
        self._logger.debug("Connected to {}".format((client.host, client.port)) +
                           (" with result code {}".format(rc) if rc != 0 else ""))

        client.publish(user_data.topic + '/available', payload='online', qos=1, retain=True)

        client.subscribe(user_data.topic + '/command', qos=1)

    def _on_disconnect(self, client, user_data, rc):
        self._logger.debug("Disconnected from {}".format((client.host, client.port)) +
                           (" with result code {}".format(rc) if rc != 0 else ""))

    def _on_message(self, client, user_data, msg):
        """Allows to start/stop the decoder, limit frame rate and enable/disable the reporting of
         detection details.

        :param client: MQTT client
        :param user_data: the record with objects to control such as decoder stop event and rate limiter
        :param msg: protocol message (command)
        """
        command = str(msg.payload, 'utf-8')
        try:
            if command.upper() == 'ON':
                # Request to turn camera on
                if user_data.stop_event.is_set():
                    self._logger.debug("Turning camera on, wait for a while...")
                    user_data.stop_event.clear()
            elif command.upper() == 'OFF':
                # Request to turn camera off
                if not user_data.stop_event.is_set():
                    self._logger.debug("Turning camera off")
                    user_data.stop_event.set()
            else:
                match = self.__command_fps.match(command)
                if match:
                    # Request to limit FPS
                    rate = match.group(1)
                    user_data.rate_limiter.limit_rate(float(rate))
                    self._logger.debug("Limiting FPS to: {}".format(rate))
                    return

                match = self.__command_details.match(command)
                if match:
                    # Request for detection details
                    b = strtobool(match.group(1))
                    with self.__command_lock:
                        self.__details = b
                    self._logger.debug("Detection details: {}".format(b))
                    return

                raise ValueError("not recognized")
        except (AssertionError, ValueError) as e:
            self._logger.error("Invalid command '{}', {}".format(command, e))
        except Exception:
            self._logger.exception('MQTT client failure')

    def _new_frame(self, frame, payload: Payload, stop_event, frame_buffer: FrameBuffer,
                   fps, decoder_fps, decoder_rate_limiter, decoder_stop_event,
                   client, user_data, *args, **kwargs):
        try:
            groups = self._group_detections_by_label(frame.header.detections)

            self._publish_states_on(client, user_data, groups)
            self._publish_states_off(client, user_data, groups)
            self._publish_detections(client, user_data, groups, frame.header.epoch)
            self._publish_sensor_info(client, user_data, frame_buffer, fps(value=True), decoder_fps())
            self._publish_state(client, user_data)
        finally:
            frame.latch.next()

    def _no_frame(self, stop_event, frame_buffer: FrameBuffer,
                  fps, decoder_fps, decoder_rate_limiter, decoder_stop_event,
                  client, user_data, *args, **kwargs):
        """Even though a frame was not received, we need to update sensors and confirm state.
        """
        self._publish_sensor_info(client, user_data, frame_buffer, fps(), decoder_fps())
        self._publish_state(client, user_data)

    @staticmethod
    def _group_detections_by_label(detections):
        groups = defaultdict(list)
        for detection in filter(lambda d: d.label > 0, detections):
            label = get_coco_class(detection.label).label

            item = {'c': round(detection.confidence * 100, 1),
                    'b': [detection.bounding_box.x_min, detection.bounding_box.y_min,
                          detection.bounding_box.x_max, detection.bounding_box.y_max]}

            zones = [z for z in filter(lambda z: z > 0, detection.zones)]
            if len(zones):
                item['z'] = zones

            groups[label].append(item)
        return groups

    def _publish_states_on(self, client, user_data, groups):
        """Publish binary state of each detected label. The state should not retain as
        it can trigger alarm state after system pause/restart even the object is no longer
        present. Instead, the state is conformed/updated every 10 seconds.
        """
        now = time()
        for label, detections in groups.items():
            reported_state = self.__states[label]
            if not reported_state.state or (now - reported_state.when) >= 10:
                reported_state.state = True
                reported_state.when = now
                client.publish(topic="{}/detection/{}/state".format(user_data.topic, label), payload='ON', qos=1,
                               retain=False)

    def _publish_states_off(self, client, user_data, groups):
        """Turn OFF state of labels, which were ON, but no longer detected.
        """
        now = time()
        for label, reported_state in self.__states.items():
            if label not in groups:
                reported_state = self.__states[label]
                if reported_state.state or (now - reported_state.when) >= 10:
                    reported_state.state = False
                    reported_state.when = now
                    client.publish(topic="{}/detection/{}/state".format(user_data.topic, label), payload='OFF', qos=1,
                                   retain=False)

    def _publish_detections(self, client, user_data, groups, epoch):
        """Publish detection details.
        """
        with self.__command_lock:
            if not self.__details:
                return

        for label, detections in groups.items():
            details = {'t': datetime.fromtimestamp(epoch).isoformat(), 'd': detections}
            client.publish(topic="{}/detection/{}/details".format(user_data.topic, label),
                           payload=json.dumps(details))

    def _publish_sensor_info(self, client, user_data, frame_buffer, fps, decoder_fps):
        """Publish information about camera sensors, when values change.
        """
        sensor = {
            'fps_in': round(decoder_fps, 1),
            'fps_out': round(fps, 1),
            'buffer': round(frame_buffer.fullness * 100)
        }
        payload = json.dumps(sensor)
        payload_hash = payload.__hash__()
        if payload_hash != self.__sensors_hash:
            self.__sensors_hash = payload_hash
            client.publish(user_data.topic + '/sensor', payload, retain=True)

    def _publish_state(self, client, user_data):
        """Publish camera state: on/off.
        """
        new_state = 'OFF' if user_data.stop_event.is_set() else 'ON'
        if new_state != self.__old_state:
            self.__old_state = new_state
            client.publish(user_data.topic + '/state', payload=new_state, qos=1, retain=True)
