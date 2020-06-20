import cv2
import numpy as np
from queue import Empty
from threading import RLock
from watsor.stream.spin import Stub
from watsor.stream.work import Work, WorkInOutPublish, Payload
from watsor.stream.share import FrameBuffer, NonSharedFramesPerSecond


class VisualEffects(WorkInOutPublish):
    """Applies video effects after inference and filtering copying the entire frame from
    the input buffer to the output buffer. Allows to subscribe other workers to consume frames
    in the output buffer.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer_in, frame_buffer_out, effects,
                 kwargs=None):
        self.__fps = NonSharedFramesPerSecond()
        super().__init__(name, stop_event, log_queue, frame_queue, frame_buffer_in, frame_buffer_out,
                         args=(effects, self.__fps), kwargs={} if kwargs is None else kwargs)

    @property
    def fps(self):
        return self.__fps

    def _incoming_frame(self, frame_in, frame_out, stop_event, effects, fps, *args, **kwargs):
        try:
            image_shape, image_np_in = frame_in.get_numpy_image(np.uint8)
            _, image_np_out = frame_out.get_numpy_image(np.uint8)
            for e in effects:
                e.apply(image_np_in, image_np_out, image_shape, frame_in.header, frame_out.header)

            fps(value=True)
        finally:
            frame_in.latch.next()


class HttpStream(Work):
    """Base class for any other HTTP streamer.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer, subscriptions,
                 args=(), kwargs=None):
        self.__stop_event = stop_event
        self.__frame_queue = frame_queue
        self.__frame_buffer = frame_buffer
        self.__subscriptions = subscriptions
        self.__args = args
        self.__started = False
        super().__init__(Stub, name, stop_event, log_queue, frame_queue,
                         kwargs={} if kwargs is None else kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Work, self)._run(stop_event, log_queue, *args, **kwargs)
        self._logger.debug("Started streaming")
        for consumer, topic in self.__subscriptions.items():
            consumer.subscribe(topic)
        self.__started = True

    def close(self):
        if self.__started:
            self.__started = False
            for consumer, topic in self.__subscriptions.items():
                consumer.unsubscribe(topic)
            self._deplete_queue(self.__frame_queue, self.__frame_buffer)
            self._logger.debug("Stopped streaming")

    @staticmethod
    def _deplete_queue(frame_queue, frame_buffer: FrameBuffer):
        """Read the queue till the end, returning frames into buffer
        """
        try:
            while True:
                payload = frame_queue.get_nowait()
                frame = frame_buffer.frames[payload.frame_index]
                frame.latch.next()
        except Empty:
            pass

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        if self.__stop_event.is_set():
            raise StopIteration()
        return self._process(self.__frame_queue, self.__stop_event, self.__frame_buffer, *self.__args)


class _JpegEncoderFrame(object):

    def __init__(self, lock=None):
        self.__lock = RLock() if lock is None else lock
        self.jpg = None
        self.epoch = 0

    @property
    def lock(self):
        return self.__lock


class MotionJpeg(HttpStream):

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer, encoder_buffer, subscriptions,
                 kwargs=None):
        super().__init__(name, stop_event, log_queue, frame_queue, frame_buffer, subscriptions,
                         args=(encoder_buffer,),
                         kwargs={} if kwargs is None else kwargs)

    @classmethod
    def create_buffer(cls, size):
        return [_JpegEncoderFrame() for _ in range(size)]

    @property
    def mime_type(self):
        return 'multipart/x-mixed-replace; boundary=--frame'

    def _next_frame(self, payload: Payload, stop_event, frame_buffer: FrameBuffer, encoder_buffer, *args,
                    **kwargs):
        frame_in = frame_buffer.frames[payload.frame_index]
        try:
            frame_out = encoder_buffer[payload.frame_index]
            with frame_out.lock:
                if frame_in.header.epoch == frame_out.epoch:
                    jpg = frame_out.jpg
                else:
                    image_shape, image_np = frame_in.get_numpy_image(np.uint8)
                    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    _, jpg = cv2.imencode('.jpg', image)

                    frame_out.jpg = jpg
                    frame_out.epoch = frame_in.header.epoch

            return self._part(jpg.tobytes())
        finally:
            frame_in.latch.next()

    def _no_frame(self, *args, **kwargs):
        return self._part(b'')

    @staticmethod
    def _part(jpg_bytes):
        msg = bytearray()
        msg.extend(b'--frame\r\n')
        msg.extend(b'Content-Type: image/jpeg\r\n')
        msg.extend(b'Content-Length: ' + f"{len(jpg_bytes)}".encode() + b'\r\n\r\n')
        msg.extend(jpg_bytes)
        msg.extend(b'\r\n')
        return bytes(msg)


class MpegTS(HttpStream):

    @property
    def mime_type(self):
        return 'video/mp2t'

    def _next_frame(self, payload: Payload, stop_event, frame_buffer: FrameBuffer, *args, **kwargs):
        frame = frame_buffer.frames[payload.frame_index]
        try:
            return bytes(frame.image.get_obj())
        finally:
            frame.latch.next()

    def _no_frame(self, *args, **kwargs):
        return b''
