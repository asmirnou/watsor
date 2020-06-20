from ctypes import memmove, memset, addressof, sizeof
from watsor.stream.share import Detection
from watsor.stream.work import WorkPassthroughPublish
from watsor.stream.share import FrameBuffer, FramesPerSecond


class DetectionSieve(WorkPassthroughPublish):
    """Filters detections right after inference, modifying detection records in frame buffer.
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer: FrameBuffer, filters,
                 decoder_rate_limiter, kwargs=None):
        self.__fps = FramesPerSecond()
        super().__init__(name, stop_event, log_queue, frame_queue, frame_buffer,
                         args=(filters, decoder_rate_limiter, self.__fps), kwargs={} if kwargs is None else kwargs)

    @property
    def fps(self):
        return self.__fps

    def _incoming_frame(self, frame, stop_event, filters, decoder_rate_limiter, fps, *args, **kwargs):
        detections = self._copy_from(frame.header.detections)
        suspicious_activity = False
        for flt in filters:
            detections, sa = flt(detections)
            suspicious_activity |= sa
        self._copy_to(frame.header.detections, detections)

        if suspicious_activity:
            if decoder_rate_limiter.unlimited():
                self._logger.debug("FPS is unlimited due to an object detected")

        fps(value=True)

    @staticmethod
    def _clone(detection):
        new_detection = Detection()
        memmove(addressof(new_detection), addressof(detection), sizeof(detection))
        return new_detection

    def _copy_from(self, detections):
        return [self._clone(detection) for detection in detections]

    @staticmethod
    def _copy_to(detections_dst, detections_src):
        iterator = iter(detections_src)
        for detection_dst in detections_dst:
            try:
                detection_src = next(iterator)
                memmove(addressof(detection_dst), addressof(detection_src), sizeof(detection_src))
            except StopIteration:
                memset(addressof(detection_dst), 0, sizeof(detection_dst))
