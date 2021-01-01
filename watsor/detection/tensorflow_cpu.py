import os
import numpy as np
import tensorflow as tf
from typing import List
from time import time
from watsor.stream.share import Detection


class TensorFlowObjectDetector:
    """Performs inference on CPU using TensorFlow.
    """

    def __init__(self, model_path):
        saved_model_path = os.path.join(model_path, 'saved_model')
        if not os.path.exists(saved_model_path):
            self.__init_tf1()
            self.__load_frozen_graph(model_path)
            return

        if int(tf.version.VERSION.split(".")[0]) == 1:
            self.__init_tf1()
            self.__load_saved_model_tf1(saved_model_path)
        else:
            self.__init_tf2()
            self.__load_saved_model_tf2(saved_model_path)
            
    @staticmethod
    def __init_tf2():
        pass

    def __init_tf1(self):
        self.__detection_graph = tf.Graph()
        self.__sess = tf.compat.v1.Session(graph=self.__detection_graph)

    def __load_saved_model_tf2(self, model_path):
        self.__serving = tf.saved_model.load(model_path,
                                             tags=[tf.saved_model.SERVING])
        if not callable(self.__serving):
            self.__serving = self.__serving.signatures['serving_default']

        self.__detect_fn = self.__detect_from_saved_model

    def __load_saved_model_tf1(self, model_path):
        tf.compat.v1.saved_model.load(self.__sess,
                                      tags=[tf.saved_model.SERVING],
                                      export_dir=model_path)

        self.__detect_fn = self.__detect_from_graph

    def __load_frozen_graph(self, model_path):
        frozen_graph_path = os.path.join(model_path, 'frozen_inference_graph.pb')
        if not os.path.isfile(frozen_graph_path):
            frozen_graph_path = os.path.join(model_path, 'cpu.pb')

        with self.__detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.__detect_fn = self.__detect_from_graph

    @property
    def device_name(self):
        return "CPU"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def detect(self, image_shape, image_np, detections: List[Detection]):
        inference_start_time = time()
        boxes, label_codes, scores = self.__detect_fn(image_np)
        inference_time = (time() - inference_start_time) * 1000

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

    def __detect_from_saved_model(self, image_np):
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, axis=0))
        output_dict = self.__serving(input_tensor)

        label_codes = output_dict['detection_classes'][0].numpy().astype(int)
        scores = output_dict['detection_scores'][0].numpy()
        boxes = output_dict['detection_boxes'][0].numpy()

        return boxes, label_codes, scores

    def __detect_from_graph(self, image_np):
        ops = self.__detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.__detection_graph.get_tensor_by_name(tensor_name)

        image_tensor = self.__detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = self.__sess.run(tensor_dict,
                                      feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        boxes = output_dict['detection_boxes'][0]
        label_codes = output_dict['detection_classes'][0]
        scores = output_dict['detection_scores'][0]

        return boxes, label_codes, scores
