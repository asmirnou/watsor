import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from operator import itemgetter
from typing import List
from time import time
from watsor.stream.share import Detection


class TensorFlowLiteObjectDetector:
    """TensorFlow Lite inference on CPU is used on IoT devices.
    TensorFlow Lite running on desktop without any GPU delegate is 9 times worse than TensorFlow.
    """

    def __init__(self, model_path):
        self.__interpreter = tflite.Interpreter(model_path=os.path.join(model_path, 'cpu.tflite'))
        self.__interpreter.allocate_tensors()

        self.__tensor_input_details = self.__interpreter.get_input_details()
        self.__tensor_output_details = self.__interpreter.get_output_details()

        self.__model_shape = itemgetter(1, 2)(self.__tensor_input_details[0]['shape'])

    @property
    def device_name(self):
        return "CPU"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def detect(self, image_shape, image_np, detections: List[Detection]):
        image_np = cv2.resize(image_np, dsize=self.__model_shape, interpolation=cv2.INTER_LINEAR)

        inference_start_time = time()
        self.__interpreter.set_tensor(self.__tensor_input_details[0]['index'],
                                      np.expand_dims(image_np, axis=0))
        self.__interpreter.invoke()

        boxes = np.squeeze(self.__interpreter.get_tensor(self.__tensor_output_details[0]['index']))
        label_codes = np.squeeze(self.__interpreter.get_tensor(self.__tensor_output_details[1]['index'])) + 1
        scores = np.squeeze(self.__interpreter.get_tensor(self.__tensor_output_details[2]['index']))

        inference_time = (time() - inference_start_time) * 1000

        d = 0
        max_width = image_shape[1] - 1
        max_height = image_shape[0] - 1
        while d < len(scores) and d < len(detections):
            detection = detections[d]
            detection.label = label_codes[d]
            detection.confidence = scores[d]
            detection.bounding_box.y_min = min(int(boxes[d][0] * max_height), max_height)
            detection.bounding_box.x_min = min(int(boxes[d][1] * max_width), max_width)
            detection.bounding_box.y_max = min(int(boxes[d][2] * max_height), max_height)
            detection.bounding_box.x_max = min(int(boxes[d][3] * max_width), max_width)
            d += 1

        return inference_time
