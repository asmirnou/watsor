from queue import Empty
from collections import namedtuple
from threading import Thread, RLock
from watsor.stream.spin import Spin
from watsor.stream.share import State, FrameBuffer
from watsor.stream.publish import Publish


class Work(Spin):
    """Basic worker receiving frames from a queue. Allows a derived class to either process a frame
    or react when nothing's received.
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, args=(), kwargs=None):
        super().__init__(delegate_class, name, stop_event, log_queue,
                         args=(frame_queue, *args), kwargs={} if kwargs is None else kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super()._run(stop_event, log_queue, *args, **kwargs)
        try:
            self._spin(self._process, stop_event, *args, **kwargs)
        except Exception:
            self._logger.exception('Spin failure')

    def _process(self, frame_queue, *args, **kwargs):
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                return
            else:
                return self._next_frame(frame, *args, **kwargs)
        except Empty:
            return self._no_frame(*args, **kwargs)

    def _no_frame(self, *args, **kwargs):
        pass

    def _next_frame(self, *args, **kwargs):
        pass


Payload = namedtuple('Payload', ['sender', 'frame_index'])


class WorkPublish(Work):
    """A worker with a shared buffer, expecting to receive an index of a frame to get from the buffer.
    Waits for PUBLISH state of a frame to let the detection complete.
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue, frame_buffer, args=(),
                 kwargs=None):
        Work.__init__(self, delegate_class, name, stop_event, log_queue, frame_queue,
                      args=(stop_event, frame_buffer, *args), kwargs={} if kwargs is None else kwargs)

    def _next_frame(self, payload: Payload, stop_event, frame_buffer: FrameBuffer, *args, **kwargs):
        frame = frame_buffer.frames[payload.frame_index]
        if not frame.latch.wait_for(State.PUBLISH, stop_event.is_set, 10):
            self._logger.error("Frame {} missed".format(payload.frame_index))
            return
        if stop_event.is_set():
            return

        self._new_frame(frame, payload, stop_event, frame_buffer, *args, **kwargs)

    def _new_frame(self, *args, **kwargs):
        args[0].latch.next()


class WorkInOutPublish(WorkPublish, Publish):
    """A worker with two frame buffers to copy a frame from in to out. Waits for PUBLISH state
    of an input frame. After processing and copying sends out the frame of the output  buffer
    to multiple subscribers. The subscribers work at PUBLISH phase as far as the worker bypasses
    DETECT state, switching a output frame from READY straight to PUBLISH.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer_in, frame_buffer_out,
                 args=(), kwargs=None):
        WorkPublish.__init__(self, Thread, name, stop_event, log_queue, frame_queue, frame_buffer_in,
                             args=(frame_buffer_out, *args),
                             kwargs={} if kwargs is None else kwargs)
        Publish.__init__(self, RLock())
        self.__last_frame_index = -1

    def _new_frame(self, frame_in, payload: Payload, stop_event, frame_buffer_in: FrameBuffer,
                   frame_buffer_out: FrameBuffer, *args, **kwargs):
        self._subscribers_lock.acquire()
        try:
            max_subscribers = len(self._subscribers)
            if max_subscribers == 0:  # Release the frame as no one has subscribed
                return

            frame_out, frame_index = frame_buffer_out.select_next_ready(self.__last_frame_index)
            self.__last_frame_index = frame_index
            if frame_out is None:
                raise BufferError

            self._incoming_frame(frame_in, frame_out, stop_event, *args, **kwargs)

            # Enforcing confirmation from all subscribers before returning frame into buffer
            frame_out.latch.next()
            frame_out.latch.next(max_subscribers)

            payload = Payload(self.name, frame_index)
            count = self._publish(payload)

            # Helps to return frame into buffer if not all subscribers received it
            while count < max_subscribers:
                frame_out.latch.next()
                count += 1
        finally:
            self._subscribers_lock.release()

    def _incoming_frame(self, *args, **kwargs):
        pass


class WorkPassthroughPublish(WorkPublish, Publish):
    """A worker supposed to be the only one waiting for PUBLISH state. Having received a frame
    it passes it through to the subscribers, expecting them to complete publishing and return
    the given frame into buffer. The given worker explicitly sets the counter of a next state
    in the frame latch, so having another worker on the same frame queue will break synchronization.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer,
                 args=(), kwargs=None):
        WorkPublish.__init__(self, Thread, name, stop_event, log_queue, frame_queue, frame_buffer,
                             args=(*args,), kwargs={} if kwargs is None else kwargs)
        Publish.__init__(self, RLock())
        self.__last_frame_index = -1

    def _new_frame(self, frame, payload: Payload, stop_event, frame_buffer: FrameBuffer, *args, **kwargs):
        self._subscribers_lock.acquire()
        try:
            max_subscribers = len(self._subscribers)
            if max_subscribers == 0:  # Release the frame as no one has subscribed
                return

            self._incoming_frame(frame, stop_event, *args, **kwargs)

            # Enforcing confirmation from all subscribers before returning frame into buffer
            frame.latch.set_next(max_subscribers)

            count = self._publish(payload)

            # Return frame into buffer if not all subscribers received it
            while count < max_subscribers:
                frame.latch.next()
                count += 1
        finally:
            self._subscribers_lock.release()

    def _incoming_frame(self, *args, **kwargs):
        pass
