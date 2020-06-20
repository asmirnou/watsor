from multiprocessing import RLock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_int, c_bool, c_double, memmove, memset, addressof, sizeof
from collections import deque, defaultdict
from logging import getLogger
from time import time
from numpy import frombuffer
from watsor.stream.sync import State, StateLatch


class BoundingBox(Structure):
    _fields_ = [('x_min', c_int),
                ('y_min', c_int),
                ('x_max', c_int),
                ('y_max', c_int)]


# noinspection PyTypeChecker
class Detection(Structure):
    _fields_ = [('label', c_int),
                ('zones', c_int * 10),
                ('confidence', c_double),
                ("bounding_box", BoundingBox)]


# noinspection PyTypeChecker
class Header(Structure):
    _fields_ = [('width', c_int),
                ('height', c_int),
                ('channels', c_int),
                ('epoch', c_double),
                ("detections", Detection * 100)]


class Frame(object):

    def __init__(self, width, height, channels, array_type_code):
        self.__lock = RLock()
        self.__header = Value(Header, width, height, channels, 0, lock=self.__lock)
        self.__image = Array(array_type_code, self.__header.width * self.__header.height * channels, lock=self.__lock)
        self.__latch = StateLatch(State.READY, self.__lock)

    def copy(self, dst):
        memmove(addressof(dst.image.get_obj()), addressof(self.__image.get_obj()), sizeof(self.__image.get_obj()))
        memmove(addressof(dst.header.get_obj()), addressof(self.__header.get_obj()), sizeof(self.__header.get_obj()))

    def clear(self):
        self.__header.epoch = 0
        memset(addressof(self.__image.get_obj()), 0, sizeof(self.__image.get_obj()))
        memset(addressof(self.__header.detections), 0, sizeof(self.__header.detections))

    @property
    def lock(self):
        return self.__lock

    @property
    def header(self):
        return self.__header

    @property
    def image(self):
        return self.__image

    @property
    def latch(self):
        return self.__latch

    def get_numpy_image(self, dtype=None):
        """# Get numpy image from buffer.
        """
        image_shape = (self.header.height, self.header.width, self.header.channels)
        image_np = frombuffer(self.image.get_obj(), dtype).reshape(image_shape)
        return image_shape, image_np


class FrameBuffer(object):

    def __init__(self, maxsize, width, height, channels=3, array_type_code='B'):
        self.__frames = []
        for d in range(maxsize):
            self.__frames.append(Frame(width, height, channels, array_type_code))

    def select_next_ready(self, start_index=-1):
        """Selects next READY frame, walking cyclically over the buffer. The search
        can start from the specified position allowing to save time iterating over
        the list of frames as previously selected frame is likely to be busy.

        :param start_index: index to start the seach
        :return: a frame with READY state
        """
        now = time()
        frame = None
        frame_index = -1
        start_index %= len(self.__frames)
        for start, end in [(start_index, len(self.__frames) - 1),
                           (-1, start_index)]:
            index = start
            while frame is None and index < end:
                index += 1
                if self.__frames[index].latch.wait(State.READY, 0):
                    frame_index = index
                    frame = self.__frames[index]
                elif self.__frames[index].header.epoch + 30 < now:
                    # If we're here something wrong's happened and
                    # we need to return this stale frame into buffer
                    frame_index = index
                    frame = self.__frames[index]
                    getLogger(self.__class__.__name__).warning(
                        "Stale frame {} dated {:.0f} seconds ago is in {}, resetting...".format(
                            index, now - frame.header.epoch, str(frame.latch.state)))
                    while not frame.latch.wait(State.READY, 0):
                        frame.latch.next()
        return frame, frame_index

    @property
    def frames(self):
        return self.__frames

    @property
    def status(self):
        status = defaultdict(int)
        for frame in self.__frames:
            status[frame.latch.state] += 1
        return status

    @property
    def fullness(self):
        return 1 - self.status[State.READY] / len(self.__frames)


class NonSharedFramesPerSecond:
    """FPS counter recording the history of events and calculating the average value within time range set.
    Calling the instance like a function returns current FPS value, clearing expired events. Setting the
    value to anything other than None, registers a new event, updating FPS counter.
    As far as this class uses deque, it can't be shared among the processes.
    """

    def __init__(self, maxlen: int = 100, time_range: float = 10.0):
        self.__timestamps = deque(maxlen=maxlen)
        self.__time_range = time_range

    def __call__(self, value=None):
        try:
            now = time()

            if value is not None:
                self.__timestamps.append(now)

            while len(self.__timestamps) > 0 and \
                    self.__timestamps[0] + self.__time_range < now:
                self.__timestamps.popleft()

            length = len(self.__timestamps)
            return length / (self.__timestamps[-1] - self.__timestamps[0]) if length > 0 else 0.0
        except ZeroDivisionError:
            return 0.0


class Cell(Structure):
    _fields_ = [('time', c_double),
                ('value', c_double)]


class FramesPerSecond(object):
    """FPS counter that can be shared among the processes as it uses shared array to keep the history
     of events rather than deque.
    """

    def __init__(self, maxlen: int = 100, timeframe: float = 10.0):
        assert maxlen > 0
        self.__lock = RLock()
        self.__timestamps = Array(Cell, [(0.0, 0.0)] * maxlen, lock=self.__lock)
        self.__index = Value('i', 0, lock=self.__lock)
        self.__start = Value('i', 0, lock=self.__lock)
        self.__length = Value('i', 0, lock=self.__lock)
        self.__maxlen = maxlen
        self.__timeframe = timeframe

    def __call__(self, value=None):
        self.__lock.acquire()
        try:
            now = time()

            if value is not None:
                self.__timestamps[self.__index.value] = (now, value)
                self.__increment(self.__index, self.__maxlen)

                if self.__length.value < self.__maxlen:
                    self.__length.value = self.__length.value + 1

                if self.__length.value == self.__maxlen:
                    self.__increment(self.__start, self.__maxlen)

            while self.__length.value > 0 and \
                    self.__timestamps[self.__start.value].time + self.__timeframe < now:
                self.__timestamps[self.__start.value] = (0, 0)
                if self.__length.value < self.__maxlen:
                    self.__increment(self.__start, self.__maxlen)
                self.__length.value = self.__length.value - 1

            if self.__length.value > 0:
                return self._calculate(self.__timestamps, self.__index.value, self.__start.value,
                                       self.__length.value, self.__maxlen)
            else:
                assert self.__start.value == self.__index.value, \
                    "start {} != index {}".format(self.__start.value, self.__index.value)
                return 0.0
        finally:
            self.__lock.release()

    @staticmethod
    def __increment(value, maxlen):
        value.value = value.value + 1
        if value.value >= maxlen:
            value.value = 0

    def _calculate(self, timestamps, index, start, length, maxlen):
        try:
            time_diff = timestamps[index - 1].time - timestamps[start].time
            return length / time_diff
        except ZeroDivisionError:
            return 0.0


class InferenceTime(FramesPerSecond):
    """Calculates the average inference time of all observations recorded in a given time range.
    Can be shared among the processes as it uses shared array to keep observation history.
    """

    def _calculate(self, timestamps, index, start, length, maxlen):
        try:
            average_value = 0.0
            for i in range(maxlen):
                average_value += timestamps[i].value
            average_value /= length
            return average_value
        except ZeroDivisionError:
            return 0.0


class RateLimiter:
    """Rate limiter based on Token Bucket algorithm.
    """

    def __init__(self):
        self.__lock = RLock()
        self.__rate = Value('d', 0, lock=self.__lock)
        self.__tokens = Value('d', 0, lock=self.__lock)
        self.__last_check = time()

    def limit_rate(self, rate):
        assert rate >= 1.0, "rate limit must be greater than 1"

        with self.__lock:
            self.__rate.value = rate
            self.__tokens.value = rate

    def unlimited(self):
        with self.__lock:
            result = self.__rate.value > 0
            self.__rate.value = 0
            self.__tokens.value = 0

        return result

    def allow(self):
        with self.__lock:
            if not self.__rate.value:
                return True

            now = time()
            time_passed = now - self.__last_check
            self.__last_check = now
            self.__tokens.value += time_passed * self.__rate.value

            if self.__tokens.value > self.__rate.value:
                self.__tokens.value = self.__rate.value

            if self.__tokens.value < 1.0:
                return False
            else:
                self.__tokens.value -= 1.0
                return True
