import os
import numpy as np
import tensorflow as tf
from typing import List
from time import time
from watsor.stream.share import Detection


class TensorFlowObjectDetector:
    """Performs inference on CPU using TensorFlow.

    Disables GPU support in TensorFlow as in favour of dedicated class TensorRTObjectDetector,
    which performs the inference more effectively:

    - Object detection on TensorFlow is memory hungry: it reserves nearly all of the GPU memory and
      requires not less than 2 GiB just for initialization.

      TensorRT performs the inference consuming only 365 MiB of GPU memory.

    - GPU support shipped with TensorFlow is compatible with limited number of CUDA-enabled devices.
      In many of cases to be able to work with GPU TensorFlow has to be rebuilt from sources.

      TensorRT runtime supports all CUDA GPUs out of the box.
    """

    def __init__(self, model_path):
        self.__detection_graph = tf.Graph()
        with self.__detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(os.path.join(model_path, 'cpu.pb'), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0}
        )
        self.__sess = tf.compat.v1.Session(
            graph=self.__detection_graph,
            config=config)

    @property
    def device_name(self):
        return "CPU"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def detect(self, image_shape, image_np, detections: List[Detection]):
        ops = self.__detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.__detection_graph.get_tensor_by_name(tensor_name)

        inference_start_time = time()
        image_tensor = self.__detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = self.__sess.run(tensor_dict,
                                      feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})
        inference_time = (time() - inference_start_time) * 1000

        boxes = output_dict['detection_boxes'][0]
        label_codes = output_dict['detection_classes'][0]
        scores = output_dict['detection_scores'][0]

        d = 0
        max_width = image_shape[1] - 1
        max_height = image_shape[0] - 1
        while d < len(scores) and d < len(detections):
            detection = detections[d]
            detection.label = label_codes[d]
            detection.confidence = scores[d]
            detection.bounding_box.y_min = int(boxes[d][0] * max_height)
            detection.bounding_box.x_min = int(boxes[d][1] * max_width)
            detection.bounding_box.y_max = int(boxes[d][2] * max_height)
            detection.bounding_box.x_max = int(boxes[d][3] * max_width)
            d += 1

        return inference_time
