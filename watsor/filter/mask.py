import cv2
from shapely.geometry import Polygon
from watsor.stream.share import Detection
from watsor.config.coco import COCO_CLASSES


class MaskFilter(object):
    """Filters only detections where bounding box intersects the opaque area in mask image.
    Mask image must be of 32 bit color and of the same resolution as corresponding camera
    video stream. Detection zone are selected in alpha channel of a mask image.

    All zones in mask image are sorted and indexed depending on the proximity of their center
    to the origin. When the bounding box detected intersects a zone, the index of that zone
    is marked in detection record.
    """

    def __init__(self, camera_config):
        # Find contours in alpha channel
        filename = camera_config['mask']
        alpha_channel, _ = get_alpha_channel(filename,
                                             camera_config['width'],
                                             camera_config['height'])
        contours = find_contours(alpha_channel)

        # Convert contours to polygons
        self.__polygons = [Polygon(c[:, 0]) for c in contours]

        # Prepare zones dictionary
        self.__polygons_by_zone = {}
        for entry in camera_config['detect']:
            coco_class = next(iter(entry))
            index = COCO_CLASSES.index(coco_class)

            zones = entry[coco_class]['zones']
            if len(zones) == 0:
                continue
            for z in zones:
                assert 0 < z <= len(self.__polygons), \
                    "There is no zone {} in mask {}".format(z, filename)

            self.__polygons_by_zone[index] = [p if idx + 1 in zones else None
                                              for idx, p in enumerate(self.__polygons)]

    def __call__(self, detection: Detection):
        bounding_box = Polygon([(detection.bounding_box.x_min, detection.bounding_box.y_min),
                                (detection.bounding_box.x_max, detection.bounding_box.y_min),
                                (detection.bounding_box.x_max, detection.bounding_box.y_max),
                                (detection.bounding_box.x_min, detection.bounding_box.y_max)])
        polygons = self.__polygons_by_zone.get(detection.label, self.__polygons)
        result = False
        z = 0
        p = 0
        while p < len(polygons) and z < len(detection.zones):
            if polygons[p] is not None and bounding_box.intersects(polygons[p]):
                detection.zones[z] = p + 1
                z += 1
                result = True
            p += 1
        return result


def get_alpha_channel(filename, width=None, height=None):
    mask_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    assert mask_image is not None, \
        "Error reading mask file {}".format(filename)

    assert len(mask_image.shape) == 3 and mask_image.shape[2] == 4, \
        "Mask image {} is not of 32 bit color".format(filename)

    if width is not None and height is not None:
        assert mask_image.shape[0] == height and mask_image.shape[1] == width, \
            "The size of mask image {} doesn't match {}x{}".format(filename, width, height)

    return mask_image[:, :, 3], mask_image


def contours_key(contour):
    moments = cv2.moments(contour)
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    return center[0] * center[0] + center[1] * center[1]


def find_contours(alpha_channel):
    ret, thresh = cv2.threshold(255 - alpha_channel, 0, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=contours_key)
    return contours
