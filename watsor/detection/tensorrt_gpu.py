import os
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import watsor.engine as engine
from collections import namedtuple
from operator import itemgetter
from pycuda.tools import clear_context_caches
from typing import List
from time import time
from watsor.stream.share import Detection


class TensorRTObjectDetector:
    """Performs object detection on Nvidia CUDA GPUs.
    TensorRT takes a trained TensorFlow network and produces a highly optimized runtime engine.
    """

    def __init__(self, model_path, device):
        cuda.init()
        self.__device = cuda.Device(device)
        self.__context = self.__device.make_context()

        trt.init_libnvinfer_plugins(engine.TRT_LOGGER, '')

        self.__trt_runtime = trt.Runtime(engine.TRT_LOGGER)
        try:
            trt_engine = engine.load_engine(self.__trt_runtime, os.path.join(model_path, 'gpu.trt'))
        except Exception as e:
            self.__finalize()
            raise e

        self.__inference = _Inference.from_engine(trt_engine)

        input_shape = trt_engine.get_binding_shape(self.__inference.input_tensor_index)
        if len(input_shape) == 4:
            input_shape = input_shape[1:]

        if input_shape[0] == 3:
            self.__transpose = True
            self.__model_shape = itemgetter(1, 2)(input_shape)  # NCHW
        elif input_shape[2] == 3:
            self.__transpose = False
            self.__model_shape = itemgetter(0, 1)(input_shape)  # NHWC

        assert self.__model_shape, "Invalid input tensor share"

    def __finalize(self):
        self.__context.pop()
        self.__context = None

        clear_context_caches()

    @property
    def device_name(self):
        return self.__device.name()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__finalize()

    def detect(self, image_shape, image_np, detections: List[Detection]):
        # Resize to model shape
        image_tensor = cv2.resize(image_np, dsize=self.__model_shape, interpolation=cv2.INTER_LINEAR)
        if self.__transpose:
            image_tensor = image_tensor.transpose((2, 0, 1))  # HWC -> CHW

        inference_start_time = time()
        boxes, classes, scores = self.__inference.detect(image_tensor)
        inference_time = (time() - inference_start_time) * 1000

        if self.__transpose:
            boxes = boxes[:, [1, 0, 3, 2]]

        d = 0
        max_width = image_shape[1] - 1
        max_height = image_shape[0] - 1
        while d < len(scores) and d < len(detections):
            detection = detections[d]
            detection.label = classes[d]
            detection.confidence = scores[d]
            detection.bounding_box.y_min = min(int(boxes[d][0] * max_height), max_height)
            detection.bounding_box.x_min = min(int(boxes[d][1] * max_width), max_width)
            detection.bounding_box.y_max = min(int(boxes[d][2] * max_height), max_height)
            detection.bounding_box.x_max = min(int(boxes[d][3] * max_width), max_width)
            d += 1

        return inference_time


_Binding = namedtuple('Binding', 'index name dtype shape allocation')


class _Inference(object):

    def __init__(self, trt_engine, input_tensor_index):
        self.__input_tensor_index = input_tensor_index
        self.__allocate_buffers(trt_engine)
        self._execution_context = trt_engine.create_execution_context()
        self.__stream = cuda.Stream()

    @property
    def input_tensor_index(self):
        return self.__input_tensor_index

    def detect(self, image_tensor):
        return [], [], []

    def __allocate_buffers(self, trt_engine):
        self.__inputs = []
        self.__outputs = []
        self.__bindings = []

        for idx, binding in enumerate(trt_engine):
            shape = trt_engine.get_binding_shape(binding)
            dtype = trt_engine.get_binding_dtype(binding)
            dtype = self._override_binding_type(binding, dtype)
            dtype_class = np.dtype(trt.nptype(dtype))

            # Allocate host and device buffers
            size = trt.volume(shape) * dtype_class.itemsize
            allocation = cuda.mem_alloc(size)

            binding_tuple = _Binding(idx, str(binding), dtype_class, list(shape), allocation)

            # Append the device buffer to device bindings.
            self.__bindings.append(allocation)

            # Append to the appropriate list.
            if trt_engine.binding_is_input(binding):
                self.__inputs.append(binding_tuple)
            else:
                self.__outputs.append(binding_tuple)

        assert len(self.__inputs) > 0
        assert len(self.__outputs) > 0
        assert len(self.__bindings) > 0

    def _override_binding_type(self, binding: str, dtype):
        return dtype

    def _do_inference(self, image_tensor, func):
        tensor_binding = self.__inputs[self.input_tensor_index]
        host_mem = np.ascontiguousarray(image_tensor.astype(dtype=tensor_binding.dtype))
        cuda.memcpy_htod_async(tensor_binding.allocation, host_mem, self.__stream)

        func(bindings=self.__bindings, stream_handle=self.__stream.handle)

        outputs = []
        for binding_tuple in self.__outputs:
            output = np.zeros(binding_tuple.shape, binding_tuple.dtype)
            cuda.memcpy_dtoh_async(output, binding_tuple.allocation, self.__stream)
            outputs.append(output)

        self.__stream.synchronize()

        return outputs

    @classmethod
    def from_engine(cls, trt_engine):
        bindings = dict((name, idx) for idx, name in enumerate(trt_engine))
        input_tensor_index = bindings.get('Input')
        if input_tensor_index is None:
            input_tensor_index = bindings.get('input_tensor')
            if input_tensor_index is None:
                assert False, "No input tensor found"
            else:
                return _InferenceONNX(trt_engine, input_tensor_index)
        else:
            return _InferenceUFF(trt_engine, input_tensor_index)


class _InferenceUFF(_Inference):

    def detect(self, image_tensor):
        # Normalize to [-1.0, 1.0] interval (expected by model)
        image_tensor = (2.0 / 255.0) * image_tensor - 1.0

        detections, _ = self._do_inference(image_tensor, self._execution_context.execute_async)

        classes = detections[0][:, 1].astype(dtype=np.int32)
        scores = detections[0][:, 2]
        boxes = detections[0][:, 3:7]

        return boxes, classes, scores

    def _override_binding_type(self, binding, dtype):
        return trt.int32 if str(binding).endswith("_1") else dtype


class _InferenceONNX(_Inference):

    def detect(self, image_tensor):
        _, boxes, scores, classes = self._do_inference(image_tensor, self._execution_context.execute_async_v2)
        return boxes[0], classes[0] + 1, scores[0]
