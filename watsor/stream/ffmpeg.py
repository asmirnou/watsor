import subprocess as sp
import threading
import logging
import io
import re
import signal
from time import time
from collections import defaultdict
from watsor.stream.read import Read, ReadPublish, ReadDetectPublish
from watsor.stream.work import Work, Payload
from watsor.stream.share import Frame, FrameBuffer, FramesPerSecond, RateLimiter
from watsor.stream.sync import CountDownLatch

try:
    SIGSTOP = signal.SIGSTOP
    SIGCONT = signal.SIGCONT
except AttributeError:  # Windows doesn't have those signals
    SIGSTOP = None
    SIGCONT = None


class FFmpegDecoder(ReadDetectPublish):
    """Controls FFmpeg subprocess (which decodes video stream), reading raw 24-bit frames and
    putting them in shared buffer. Triggers detection and allows other workers to subscribe
    to perform their work for when detection and filtering are complete.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer, cmd_args, cwd=None,
                 stdin=sp.DEVNULL, kwargs=None):
        self.__cmd_args = cmd_args
        self.__cwd = cwd
        self.__stdin = stdin
        self.__subprocess = None
        self.__stderr_thread = None
        self.__fps = FramesPerSecond()
        self.__rate_limiter = RateLimiter()
        self.__info_latch = CountDownLatch() if any('showinfo' in s for s in cmd_args) else None
        self.__info_re = re.compile(r"(\w+):\s*((?:\[[^\]]+\])|(?:[^\s]+))", re.IGNORECASE)
        super().__init__(name, stop_event, log_queue, frame_queue, frame_buffer,
                         args=(self.__fps, self.__rate_limiter), kwargs={} if kwargs is None else kwargs)

    def initialize(self):
        assert self.__subprocess is None or self.__subprocess.poll() is not None, \
            "Subprocess has not terminated yet"

        self.__subprocess = sp.Popen(args=self.__cmd_args, cwd=self.__cwd,
                                     stdout=sp.PIPE, stderr=sp.PIPE, stdin=self.__stdin)
        if SIGSTOP is not None:
            self.__subprocess.send_signal(signal.SIGSTOP)

        self.__stderr_thread = threading.Thread(name=self.name, target=_stderr_reader,
                                                args=(self.__class__.__name__,
                                                      self.__subprocess.stderr,
                                                      self._parse_show_info))
        self.__stderr_thread.daemon = True

        super().initialize()

    @property
    def stdin(self):
        return self.__subprocess.stdin

    @property
    def fps(self):
        return self.__fps

    @property
    def rate_limiter(self):
        return self.__rate_limiter

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Read, self)._run(stop_event, log_queue, *args, **kwargs)
        self._logger.debug(self.__cmd_args)
        self._logger.debug("Reading from stdout")
        try:
            self._spin(self._process, stop_event, *args, **kwargs)
        except EOFError:
            pass  # Ignore when end of stream (file) reached and stop gracefully
        except Exception:
            self._logger.exception('Spin failure')
        finally:
            self.__subprocess.stdout.close()
            self._logger.debug("Stdout closed")

    def _new_frame(self, frame: Frame, frame_queue, frame_buffer, fps, rate_limiter, *args, **kwargs):
        if self.__info_latch is not None:
            self.__info_latch.reset(1)

        frame.clear()
        image = self.__subprocess.stdout.readinto(frame.image.get_obj())
        if not image:
            raise EOFError
        frame.header.epoch = time()

        if not rate_limiter.allow():
            return False

        fps(value=True)

        if self.__info_latch is not None and self.__info_latch.wait(1):
            frame.header.info.has = True
            frame.header.info.num = int(self.__info['n'])
            frame.header.info.pts = int(self.__info['pts'])
            frame.header.info.pts_time = float(self.__info['pts_time'])

        return True

    def start(self):
        super().start()
        self.__stderr_thread.start()
        if SIGCONT is not None:
            self.__subprocess.send_signal(signal.SIGCONT)

    def terminate(self):
        if SIGCONT is not None:
            self.__subprocess.send_signal(signal.SIGCONT)
        self.__subprocess.terminate()
        super().terminate()

    def join(self, timeout=None):
        try:
            super().join(timeout)
            self.__subprocess.wait(timeout)
        except Exception:
            self.__subprocess.terminate()
            raise
        finally:
            self.__subprocess.stderr.close()

    def _parse_show_info(self, logger, line):
        if line.startswith("[Parsed_showinfo_"):
            logger.debug(line)
        else:
            logger.info(line)
            return

        if self.__info_latch is None or self.__info_latch.wait(0):
            return

        self.__info = defaultdict(list)
        for m in self.__info_re.finditer(line):
            self.__info[m.group(1)] = m.group(2)

        if 'n' in self.__info:
            self.__info_latch.count_down()


class FFmpegEncoder(Work):
    """Controls FFmpeg subprocess feeding it with raw 24-bit frames to encode them
     in compressed video stream. Exposes FFmpeg subprocess stdout for further re-streaming.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer, cmd_args, cwd=None,
                 stdout=sp.DEVNULL,
                 args=(), kwargs=None):
        self.__cmd_args = cmd_args
        self.__cwd = cwd
        self.__stdout = stdout
        self.__subprocess = None
        self.__stderr_thread = None
        self.__fps = FramesPerSecond()
        self.__written = None
        super().__init__(threading.Thread, name, stop_event, log_queue, frame_queue,
                         args=(stop_event, frame_buffer, self.__fps, *args),
                         kwargs={} if kwargs is None else kwargs)

    def initialize(self):
        assert self.__subprocess is None or self.__subprocess.poll() is not None, \
            "Subprocess has not terminated yet"

        self.__subprocess = sp.Popen(args=self.__cmd_args, cwd=self.__cwd,
                                     stdout=self.__stdout, stderr=sp.PIPE, stdin=sp.PIPE)
        if SIGSTOP is not None:
            self.__subprocess.send_signal(signal.SIGSTOP)

        self.__stderr_thread = threading.Thread(name=self.name, target=_stderr_reader,
                                                args=(self.__class__.__name__,
                                                      self.__subprocess.stderr,
                                                      _default_printer))
        self.__stderr_thread.daemon = False

        super().initialize()

    @property
    def stdout(self):
        return self.__subprocess.stdout

    @property
    def fps(self):
        return self.__fps

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Work, self)._run(stop_event, log_queue, *args, **kwargs)
        self._logger.debug(self.__cmd_args)
        self._logger.debug("Writing to stdin")
        try:
            self._spin(self._process, stop_event, *args, **kwargs)
        except BrokenPipeError:
            pass  # Ignore broken pipe errors, as ffmpeg exited before all data were written.
        except Exception:
            self._logger.exception('Spin failure')
        finally:
            self._close_stdin()
            self._logger.debug("Stdin closed")

    def _next_frame(self, payload: Payload, stop_event, frame_buffer: FrameBuffer, fps, *args, **kwargs):
        frame = frame_buffer.frames[payload.frame_index]
        try:
            self.__written = self.__subprocess.stdin.write(frame.image.get_obj())
            self.__subprocess.stdin.flush()

            fps(value=True)
        finally:
            frame.latch.next()

    def _close_stdin(self):
        try:
            self.__subprocess.stdin.close()
        except BrokenPipeError:
            pass  # Ignore broken pipe errors, as program exited before all data were written.

        if self.__written is None:
            # Interrupt waiting FFmpeg to let it finish stream without any data written to it.
            self.__subprocess.send_signal(signal.SIGINT)

    def start(self):
        super().start()
        self.__stderr_thread.start()
        if SIGCONT is not None:
            self.__subprocess.send_signal(signal.SIGCONT)

    def terminate(self):
        if SIGCONT is not None:
            self.__subprocess.send_signal(signal.SIGCONT)
        self.__subprocess.terminate()
        super().terminate()

    def join(self, timeout=None):
        try:
            super().join(timeout)
            self.__subprocess.wait(timeout)
        except Exception:
            self.__subprocess.terminate()
            raise
        finally:
            self.__subprocess.stderr.close()


def _stderr_reader(log_name, stream, consumer):
    wrapper = io.TextIOWrapper(stream)
    logger = logging.getLogger(log_name)
    logger.debug("Stderr redirected to stdout")
    try:
        line = wrapper.readline()
        while line:
            consumer(logger, line.rstrip())
            line = wrapper.readline()
    except Exception as e:
        logger.exception(e)
    finally:
        wrapper.close()
    logger.debug("Stderr gracefully closed")


def _default_printer(logger, line):
    logger.info(line)


class MpegTSReader(ReadPublish):
    """Reads encoded (compressed) video stream from FFmpeg subprocess by chunks equal to the size of
    a frame in the buffer. The buffer is filled with 188-byte sections of MPEG-TS stream, where each
    frame contains many of those packets, not necessarily the whole.
    Allows other workers to subscribe in order to re-stream the packets from the buffer.
    """

    def __init__(self, name: str, stop_event, log_queue, stream, frame_buffer, kwargs=None):
        super().__init__(name, stop_event, log_queue, stream, frame_buffer,
                         kwargs={} if kwargs is None else kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Read, self)._run(stop_event, log_queue, *args, **kwargs)
        self._logger.debug("Reading from stdout")
        try:
            self._spin(self._process, stop_event, *args, **kwargs)
        except EOFError:
            pass  # Ignore when end of stream (file) reached and stop gracefully
        except Exception:
            self._logger.exception('Spin failure')
        finally:
            self._close(*args, **kwargs)
            self._logger.debug("Stdout closed")

    @staticmethod
    def _close(stream, *args, **kwargs):
        stream.close()

    def _new_frame(self, frame: Frame, stream, *args, **kwargs):
        frame.clear()
        image = stream.readinto(frame.image.get_obj())
        if not image:
            raise EOFError
        frame.header.epoch = time()
        return True
