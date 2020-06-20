from time import time
from collections import namedtuple, defaultdict
from multiprocessing import Value
from watsor.stream.spin import Spin
from watsor.stream.read import Read
from watsor.stream.work import Work

Frame = namedtuple('Frame', ['device', 'time'])


class DummyRead(Read):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, fps):
        self._fps = fps
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue,
                         args=(stop_event,))

    def _next_frame(self, frame_queue, stop_event):
        if stop_event.wait(1 / self._fps):
            return None

        return Frame(self.name, time())

    @property
    def fps(self):
        return self._fps


class DummyWork(Work):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, done_queue, fps):
        self._fps = fps
        self._last_frame_time = float('inf')
        self._max_lag = Value('d', 0)
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue,
                         args=(stop_event, self._max_lag),
                         kwargs={'done_queue': done_queue})

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Work, self)._run(stop_event, log_queue, *args, **kwargs)
        try:
            count = defaultdict(int)
            self._spin(self._process, stop_event, *args, count)
            self._done(count, **kwargs)
        except Exception:
            self._logger.exception('Run failure')

    def _next_frame(self, frame: Frame, stop_event, max_lag, count):
        max_lag.value = max(frame.time - self._last_frame_time, max_lag.value)
        self._last_frame_time = frame.time

        count[frame.device] += 1  # Count the number of frames produced by each reader

        stop_event.wait(1 / self._fps)

    def _done(self, count, done_queue):
        # Report the readers distribution data
        for key in count:
            done_queue.put_nowait((key, count[key]))
        # Report the number of frames processed by this worker
        done_queue.put_nowait((self.name, sum(count.values())))

    @property
    def fps(self):
        return self._fps

    @property
    def max_lag(self):
        return self._max_lag.value


class Stumble(Spin):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, interval, count_down_latch, kwargs=None):
        super().__init__(delegate_class, name, stop_event, log_queue,
                         args=(interval, count_down_latch),
                         kwargs={} if kwargs is None else kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super()._run(stop_event, log_queue, *args, **kwargs)
        self._wait(stop_event, *args, **kwargs)
        self._logger.debug("Exited")

    def _wait(self, stop_event, interval, count_down_latch, *args, **kwargs):
        count_down_latch.count_down()
        self._logger.debug("Started, waiting for {} sec".format(interval))
        stop_event.wait(interval)
