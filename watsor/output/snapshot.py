import cv2
import numpy as np
from threading import Thread
from collections import namedtuple, defaultdict
from watsor.config.coco import COCO_CLASSES
from watsor.stream.work import WorkPublish
from watsor.stream.share import NonSharedFramesPerSecond
from watsor.stream.share import FrameBuffer

KeepData = namedtuple('userdata', ['frame_index', 'confidence', 'last_update'])


class Snapshot(WorkPublish):
    """Remembers the last detection of an object class. A subsequent detection with greater
    confidence replaces the current snapshot. A detection with lower confidence will replace
    the current snapshot after its relevance expires (10 seconds).
    """

    def __init__(self, name: str, stop_event, log_queue, frame_queue, frame_buffer, camera_config,
                 effects=None, expire=10, kwargs=None):
        self.__fps = NonSharedFramesPerSecond()
        self.__effects = [] if effects is None else effects
        self.__init_frame_keeper(camera_config)
        super().__init__(Thread, name, stop_event, log_queue, frame_queue, frame_buffer,
                         args=(expire, self.__fps), kwargs={} if kwargs is None else kwargs)

    def __init_frame_keeper(self, camera_config):
        self.__dict = {}
        frame_index = 0
        for entry in camera_config['detect']:
            coco_class = next(iter(entry))
            idx = COCO_CLASSES.index(coco_class)
            self.__dict[idx] = KeepData(frame_index, 0, 0)
            frame_index += 1

        self.__frame_keeper = FrameBuffer(frame_index, camera_config['width'], camera_config['height'])

    @property
    def fps(self):
        return self.__fps

    def get(self, coco_class, extension='.jpg'):
        assert coco_class in COCO_CLASSES, "Unknown object class '{}'".format(coco_class)
        label = COCO_CLASSES.index(coco_class)
        assert label in self.__dict, "Object class '{}' is not configured for detection".format(coco_class)
        keep_data = self.__dict[label]
        frame = self.__frame_keeper.frames[keep_data.frame_index]

        # Leave only detections for the given label
        for detection in frame.header.detections:
            if detection.label != label:
                detection.label = 0

        # Apply effects
        image_shape, image_np_in = frame.get_numpy_image(np.uint8)
        image_np_out = np.array(image_np_in)
        for e in self.__effects:
            e.apply(image_np_in, image_np_out, image_shape, frame.header, frame.header)

        # Encode image
        image = cv2.cvtColor(image_np_out, cv2.COLOR_BGR2RGB)
        return cv2.imencode(extension, image)

    def _new_frame(self, frame, payload, stop_event, frame_buffer, expire, fps, *args, **kwargs):
        try:
            groups = self._select_most_confident(frame.header.detections)

            for label, confidence in groups.items():
                try:
                    keep_data = self.__dict[label]
                except KeyError:
                    continue

                if confidence <= keep_data.confidence and \
                        frame.header.epoch - keep_data.last_update <= expire:
                    continue

                frame_to_keep = self.__frame_keeper.frames[keep_data.frame_index]
                frame.copy(frame_to_keep)

                self.__dict[label] = KeepData(keep_data.frame_index, confidence, frame.header.epoch)

            fps(value=True)
        finally:
            frame.latch.next()

    @staticmethod
    def _select_most_confident(detections):
        groups = defaultdict(int)
        for detection in filter(lambda d: d.label > 0, detections):
            if detection.confidence > groups[detection.label]:
                groups[detection.label] = detection.confidence
        return groups
