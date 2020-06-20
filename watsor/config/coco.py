import numpy as np
from collections import namedtuple

CocoClass = namedtuple('CocoClass', [
    'label',
    'box_color',
    'font_color',
    'box_thickness',
    'font_thickness',
    'font_scale',
    'alpha',
])

COCO_CLASSES = [
    'unlabeled',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

# Random RGB colors for each label
_COCO_COLORS = np.random.RandomState(255).uniform(0, 255, size=(len(COCO_CLASSES), 3)).astype(np.uint8)

# Dictionary of labels with index as a key
_COCO_DICTIONARY = {
    idx: CocoClass(label,
                   box_color=tuple(map(int, _COCO_COLORS[idx])),
                   font_color=(255, 255, 255),
                   box_thickness=1,
                   font_thickness=1,
                   font_scale=0.5,
                   alpha=0.55)
    for idx, label in enumerate(COCO_CLASSES)
}


def get_coco_class(idx):
    """Finds COCO class in build-in dictionary, returning 'unlabeled' if not found.

    :param idx: index of COCO class to find
    :return: CocoClass instance
    """

    return _COCO_DICTIONARY.get(idx, _COCO_DICTIONARY[0])
