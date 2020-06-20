from watsor.config.coco import COCO_CLASSES
from watsor.stream.share import Detection


class ConfidenceFilter(object):
    """Filters only detections with a confidence greater than the threshold set.
    Threshold is set per label (detection object class).
    """

    def __init__(self, camera_config):
        self.__indexes = {}
        for entry in camera_config['detect']:
            coco_class = next(iter(entry))
            idx = COCO_CLASSES.index(coco_class)
            self.__indexes[idx] = entry[coco_class]['confidence'] / 100

    def __call__(self, detection: Detection):
        confidence = self.__indexes.get(detection.label, None)
        return confidence is not None and detection.confidence >= confidence
