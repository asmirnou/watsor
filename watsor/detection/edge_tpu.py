import os
import cv2
from typing import List
from operator import itemgetter
from edgetpu.detection.engine import DetectionEngine
from watsor.stream.share import Detection


class CoralObjectDetector:
    """Performs inference on Edge TPU.
    """

    def __init__(self, model_path, device_path):
        self.__engine = DetectionEngine(model_path=os.path.join(model_path, 'edgetpu.tflite'),
                                        device_path=device_path)

        self.__model_shape = itemgetter(1, 2)(self.__engine.get_input_tensor_shape())

    @property
    def device_name(self):
        return "Coral"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def detect(self, image_shape, image_np, detections: List[Detection]):
        image_np = cv2.resize(image_np, dsize=self.__model_shape, interpolation=cv2.INTER_LINEAR)

        objs = self.__engine.detect_with_input_tensor(input_tensor=image_np.flatten(),
                                                      top_k=len(detections))

        d = 0
        max_width = image_shape[1] - 1
        max_height = image_shape[0] - 1
        while d < len(objs) and d < len(detections):
            detection = detections[d]
            obj = objs[d]
            detection.label = obj.label_id + 1
            detection.confidence = obj.score
            detection.bounding_box.y_min = int(obj.bounding_box[0][1] * max_height)
            detection.bounding_box.x_min = int(obj.bounding_box[0][0] * max_width)
            detection.bounding_box.y_max = int(obj.bounding_box[1][1] * max_height)
            detection.bounding_box.x_max = int(obj.bounding_box[1][0] * max_width)
            d += 1

        return self.__engine.get_inference_time()
