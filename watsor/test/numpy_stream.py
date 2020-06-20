import random
import numpy as np
from time import time
from collections import namedtuple
from multiprocessing import Value
from watsor.stream.read import Read
from watsor.stream.work import Work
from watsor.stream.share import FrameBuffer

Action = namedtuple('Action', ['frame_index', 'operation', 'value', 'expected_result'])

OPERATIONS = "*/+-"


def simple_math(operation, x, y):
    return {
        OPERATIONS[0]: lambda _x: _x * y,
        OPERATIONS[1]: lambda _x: _x / y,
        OPERATIONS[2]: lambda _x: _x + y,
        OPERATIONS[3]: lambda _x: _x - y
    }[operation](x)


class NumpyRead(Read):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer):
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue,
                         args=(stop_event, frame_buffer))
        self._last_frame_index = -1

    def _next_frame(self, frame_queue, stop_event, frame_buffer: FrameBuffer):
        frame, frame_index = frame_buffer.select_next_ready(self._last_frame_index)
        self._last_frame_index = frame_index
        if frame is None:
            return None

        image_shape, image_np = frame.get_numpy_image()

        # Fill image buffer with random data
        data = np.random.randn(*image_shape)
        np.copyto(image_np, data)

        # Calculate sum of all pixels and do math against the sum
        x = np.sum(image_np[:]).item()
        y = random.random()
        operation = OPERATIONS[random.randint(0, len(OPERATIONS[0]))]
        expected_result = simple_math(operation, x, y)

        frame.header.epoch = time()
        frame.latch.next()

        return Action(frame_index, operation, y, expected_result)


class NumpyWork(Work):

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer):
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue,
                         args=(stop_event, frame_buffer))
        self._matches = Value('i', 0)

    def _next_frame(self, action: Action, stop_event, frame_buffer: FrameBuffer):
        frame = frame_buffer.frames[action.frame_index]
        try:
            # Get image numpy buffer
            image_shape = (frame.header.height, frame.header.width, frame.header.channels)
            image_np = np.frombuffer(frame.image.get_obj()).reshape(image_shape)

            # Do math against numpy image, changing the raw data
            simple_math(action.operation, image_np, action.value)

            # Calculate sum of all pixels and compare with expected value
            x = np.sum(image_np[:]).item()
            result = simple_math(action.operation, x, action.value)
            if round(result - action.expected_result, 1) == 0:
                # Count position match
                self._matches.value = self._matches.value + 1
        finally:
            frame.latch.next()

    @property
    def matches(self):
        return self._matches.value
