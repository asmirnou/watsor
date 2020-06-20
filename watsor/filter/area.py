from watsor.config.coco import COCO_CLASSES
from watsor.stream.share import Detection, BoundingBox


class AreaFilter(object):
    """Filters only detections with an area of a bounding box greater than the threshold set.
    Threshold is set per label (detection object class).
    """

    def __init__(self, camera_config):
        self.__indexes = {}
        for entry in camera_config['detect']:
            coco_class = next(iter(entry))
            idx = COCO_CLASSES.index(coco_class)
            width = camera_config['width']
            height = camera_config['height']
            max_area = self.area(BoundingBox(0, 0, width - 1, height - 1))
            self.__indexes[idx] = entry[coco_class]['area'] / 100 * max_area

    def __call__(self, detection: Detection):
        area = self.__indexes.get(detection.label, None)
        return area is not None and self.area(detection.bounding_box) >= area

    @staticmethod
    def area(bb: BoundingBox):
        return abs((bb.x_max - bb.x_min + 1) * (bb.y_max - bb.y_min + 1))
