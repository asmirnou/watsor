from queue import Full
from threading import Thread, RLock
from watsor.stream.spin import Spin
from watsor.stream.publish import Publish
from watsor.stream.sync import State
from watsor.stream.share import FrameBuffer
from watsor.stream.work import Payload


class Read(Spin):
    """Basic spinner reading the frames from a source and sending them out through a queue.
    If a queue is full, the frame is dropped and next one is read. The derived class must
    override _next_frame method to generate a new frame.
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, args=(), kwargs=None):
        super().__init__(delegate_class, name, stop_event, log_queue,
                         args=(frame_queue, *args,), kwargs={} if kwargs is None else kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super()._run(stop_event, log_queue, *args, **kwargs)
        try:
            self._spin(self._process, stop_event, *args, **kwargs)
        except Exception:
            self._logger.exception('Spin failure')

    def _process(self, *args, **kwargs):
        frame = self._next_frame(*args, **kwargs)
        if frame is None:
            return
        self._send_frame(frame, *args, **kwargs)

    def _next_frame(self, *args, **kwargs):
        return None

    def _send_frame(self, frame, frame_queue, *args, **kwargs):
        try:
            frame_queue.put_nowait(frame)
        except Full:
            pass


class ReadFrameBuffer(Read):
    """A reader with a shared buffer to place frames in. The workers are supposed to share the same
    buffer, so the reader transmits through the queue only frame index to notify the parties that
    the frame can be processed.
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer,
                 args=(), kwargs=None):
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue,
                         args=(frame_buffer, *args), kwargs={} if kwargs is None else kwargs)
        self.__last_frame_index = -1

    def _next_frame(self, frame_queue, frame_buffer: FrameBuffer, *args, **kwargs):
        frame, frame_index = frame_buffer.select_next_ready(self.__last_frame_index)
        self.__last_frame_index = frame_index
        if frame is None:
            raise BufferError

        return frame_index if self._new_frame(frame, frame_queue, frame_buffer, *args, **kwargs) else None

    def _new_frame(self, *args, **kwargs):
        pass


class ReadPublish(ReadFrameBuffer, Publish):
    """A reader with a shared frame buffer and an ability to send out a frame to multiple subscribers.
    Doesn't use frame queue of the parent class, but sends out to the queues of subscribers. Supposed
    to be used bypassing DETECT state, switching a frame from READY straight to PUBLISH.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer, args=(), kwargs=None):
        ReadFrameBuffer.__init__(self, Thread, name, stop_event, log_queue, frame_queue, frame_buffer,
                                 args=(*args,), kwargs={} if kwargs is None else kwargs)
        Publish.__init__(self, RLock())

    def _send_frame(self, frame_index, frame_queue, *args, **kwargs):
        self._multi_put_frame(frame_index, frame_queue, *args, **kwargs)

    def _multi_put_frame(self, frame_index, frame_queue, frame_buffer: FrameBuffer, *args, **kwargs):
        self._subscribers_lock.acquire()
        try:
            max_subscribers = len(self._subscribers)
            if max_subscribers == 0:
                return

            frame = frame_buffer.frames[frame_index]
            payload = Payload(self.name, frame_index)

            # Enforcing confirmation from all subscribers before returning frame into buffer
            frame.latch.next()
            frame.latch.next(max_subscribers)

            count = self._publish(payload)

            # Helps to return frame into buffer if not all subscribers received it
            while count < max_subscribers:
                frame.latch.next()
                count += 1
        finally:
            self._subscribers_lock.release()


class ReadDetectPublish(ReadFrameBuffer, Publish):
    """A reader with a shared frame buffer and an ability to send out a frame to multiple subscribers.
    Uses frame queue of the parent class to deliver a frame to all registered object detectors first.
    After that sends out the given frame to the subscribers, counting the number acknowledged the delivery.
    The subscribers suppose to wait will detector completes before starting to work on a frame.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer, args=(), kwargs=None):
        ReadFrameBuffer.__init__(self, Thread, name, stop_event, log_queue, frame_queue, frame_buffer,
                                 args=(*args,), kwargs={} if kwargs is None else kwargs)
        Publish.__init__(self, RLock())

    def _send_frame(self, frame_index, frame_queue, *args, **kwargs):
        self._multi_put_frame(frame_index, frame_queue, *args, **kwargs)

    def _multi_put_frame(self, frame_index, frame_queue, frame_buffer: FrameBuffer, *args, **kwargs):
        self._subscribers_lock.acquire()
        try:
            max_subscribers = len(self._subscribers)
            if max_subscribers == 0:  # Release the frame as no one has subscribed
                return

            frame = frame_buffer.frames[frame_index]
            payload = Payload(self.name, frame_index)

            # Enforcing two conditions to happen before switching the state of the frame to PUBLISH:
            # - detection complete
            # - publishing to available subscribers complete
            frame.latch.next(2)

            # Put frame into detection queue
            frame_queue.put_nowait(payload)

            # Publish frame to subscribers
            count = self._publish(payload)

            # Enforcing confirmation from all subscribers before moving to PUBLISH state
            frame.latch.next(count)

            # Return frame into buffer as no one subscribed
            if count == 0:
                frame.latch.next()
        except Full:
            # Return frame into buffer by traversing through states as frame queue is full
            while not frame.latch.wait(State.READY, 0):
                frame.latch.next()
        finally:
            self._subscribers_lock.release()
