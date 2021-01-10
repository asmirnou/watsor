import os
import cv2
import numpy as np
from time import time
from typing import List
from operator import itemgetter
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from watsor.stream.share import Detection


class CoralObjectDetector:
    """Performs inference on Edge TPU.
    """

    def __init__(self, model_path, device):
        self.__interpreter = edgetpu.make_interpreter(os.path.join(model_path, 'edgetpu.tflite'),
                                                      device=device)
        self.__interpreter.allocate_tensors()

        self.__model_shape = common.input_size(self.__interpreter)

    @property
    def device_name(self):
        return "Coral"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def detect(self, image_shape, image_np, detections: List[Detection]):
        image_np = cv2.resize(image_np, dsize=self.__model_shape, interpolation=cv2.INTER_LINEAR)
        limits = np.subtract(itemgetter(1, 0)(image_shape), (1, 1))
        image_scale = np.divide(self.__model_shape, limits)

        inference_start_time = time()
        common.set_input(self.__interpreter, image_np)
        self.__interpreter.invoke()
        objs = detect.get_objects(self.__interpreter, image_scale=image_scale)
        inference_time = (time() - inference_start_time) * 1000

        d = 0
        while d < len(objs) and d < len(detections):
            detection = detections[d]
            obj = objs[d]
            detection.label = obj.id + 1
            detection.confidence = obj.score
            detection.bounding_box.y_min = min(obj.bbox.ymin, limits[1])
            detection.bounding_box.x_min = min(obj.bbox.xmin, limits[0])
            detection.bounding_box.y_max = min(obj.bbox.ymax, limits[1])
            detection.bounding_box.x_max = min(obj.bbox.xmax, limits[0])
            d += 1

        return inference_time
