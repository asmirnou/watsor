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

HostDeviceMem = namedtuple('HostDeviceMem', 'host device')


class TensorRTObjectDetector:
    """Performs object detection on Nvidia CUDA GPUs.
    TensorRT takes a trained TensorFlow network and produces a highly optimized runtime engine.
    """

    def __init__(self, model_path, device):
        cuda.init()
        self.__device = cuda.Device(device)
        self.context = self.__device.make_context()

        engine.load_plugins()

        self.__trt_runtime = trt.Runtime(engine.TRT_LOGGER)
        self.__trt_engine = engine.load_engine(self.__trt_runtime, os.path.join(model_path, 'gpu.buf'))

        self._allocate_buffers()

        self.__model_shape = itemgetter(1, 2)(self.__trt_engine.get_binding_shape('Input'))
        self.__execution_context = self.__trt_engine.create_execution_context()

    @property
    def device_name(self):
        return self.__device.name()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.context.pop()
        self.context = None

        clear_context_caches()

    def detect(self, image_shape, image_np, detections: List[Detection]):
        # Resize to model shape
        image_tensor = cv2.resize(image_np, dsize=self.__model_shape, interpolation=cv2.INTER_LINEAR)
        # HWC -> CHW
        image_tensor = image_tensor.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        image_tensor = (2.0 / 255.0) * image_tensor - 1.0
        image_tensor = image_tensor.ravel()

        np.copyto(self.inputs[0].host, image_tensor)
        inference_start_time = time()
        detection_out, keep_count_out = self._do_inference()
        inference_time = (time() - inference_start_time) * 1000

        d = 0
        max_width = image_shape[1] - 1
        max_height = image_shape[0] - 1
        while d < int(keep_count_out[0]) and d < len(detections):
            detection = detections[d]
            pred_start_idx = d * 7
            detection.label = detection_out[pred_start_idx + 1]
            detection.confidence = detection_out[pred_start_idx + 2]
            detection.bounding_box.x_min = int(detection_out[pred_start_idx + 3] * max_width)
            detection.bounding_box.y_min = int(detection_out[pred_start_idx + 4] * max_height)
            detection.bounding_box.x_max = int(detection_out[pred_start_idx + 5] * max_width)
            detection.bounding_box.y_max = int(detection_out[pred_start_idx + 6] * max_height)
            d += 1

        return inference_time

    def _do_inference(self):
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.__execution_context.execute_async(batch_size=self.__trt_engine.max_batch_size,
                                               bindings=self.bindings,
                                               stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        return [out.host for out in self.outputs]

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # NMS implementation in TRT 6 only supports DataType.FLOAT
        binding_to_type = {"Input": np.float32,
                           "NMS": np.float32,
                           "NMS_1": np.int32}
        for binding in self.__trt_engine:
            shape = self.__trt_engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.__trt_engine.max_batch_size
            dtype = binding_to_type[str(binding)]

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))

            # Append to the appropriate list.
            if self.__trt_engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
