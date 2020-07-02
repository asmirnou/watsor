from numpy import uint8
from typing import Dict
from multiprocessing.sharedctypes import Array
from watsor.stream.work import Work, Payload
from watsor.stream.share import FrameBuffer, FramesPerSecond, InferenceTime
from watsor.detection.devices import cuda_gpus, edge_tpus, cpus

_ALWAYS_USE_CPU = False


def create_object_detectors(delegate_class, stop_event, log_queue, frame_queue, frame_buffers, model_path, kwargs=None):
    """Creates all available detectors.
    Falls back to CPU if neither GPU or Coral detected.

    :param delegate_class: either Thread of Process
    :param stop_event: event to stop the detection process
    :param log_queue: queue to put log messages
    :param frame_queue: queue to wait for a frame to process
    :param frame_buffers: dictionary of frame buffers. A frame payload contains
                          `sender` key to find a buffer in the dictionary,
    :param model_path: path where the NN model is
    :param kwargs: dictionary with options such as log level
    :return: list of detector processes to start
    """

    detectors = []
    if kwargs is None:
        kwargs = {}

    def gen_name():
        return "detector{}".format(len(detectors) + 1)

    def append_detector(*args):
        detectors.append(ObjectDetector(delegate_class, gen_name(), stop_event, log_queue, frame_queue, frame_buffers,
                                        kwargs={**kwargs,
                                                'detector_class': clazz,
                                                'detector_args': (model_path, *args,)}))

    for device, clazz in edge_tpus():
        append_detector(device)

    for device, clazz in cuda_gpus():
        append_detector(device)

    if _ALWAYS_USE_CPU or len(detectors) == 0:
        for clazz in cpus():
            append_detector()

    assert len(detectors) > 0, "Failed to create an object detector. Make sure TensorFlow is installed."

    return detectors


class ObjectDetector(Work):
    """Performs object detection inference without modifying image data,
    but filling detection records.
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, frame_queue_in,
                 frame_buffers: Dict[object, FrameBuffer], kwargs=None):
        self.__fps = FramesPerSecond()
        self.__inference_time = InferenceTime()
        self.__device_name = Array('c', 255)
        super().__init__(delegate_class, name, stop_event, log_queue, frame_queue_in,
                         args=(stop_event, frame_buffers, self.__fps, self.__inference_time),
                         kwargs={} if kwargs is None else kwargs)

    @property
    def device_name(self):
        return self.__device_name.value

    @property
    def fps(self):
        return self.__fps

    @property
    def inference_time(self):
        return self.__inference_time

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super(Work, self)._run(stop_event, log_queue, *args, **kwargs)
        try:
            detector_class = kwargs.get('detector_class')
            detector_args = kwargs.get('detector_args')

            with detector_class(*detector_args) as object_detector:
                self.__device_name.value = str.encode(object_detector.device_name)[:len(self.__device_name)]

                self._logger.debug("{}{} initialized".format(
                    object_detector.__class__.__name__, detector_args))

                self._spin(self._process, stop_event, *args, object_detector, **kwargs)
        except FileNotFoundError as e:
            self._logger.error(e)
        except Exception:
            self._logger.exception('Detection failure')

    def _next_frame(self, payload: Payload, stop_event, frame_buffers, fps, inference_time,
                    object_detector, *args, **kwargs):
        frame = frame_buffers[payload.sender].frames[payload.frame_index]
        try:
            image_shape, image_np = frame.get_numpy_image(uint8)
            time_of_inference = object_detector.detect(image_shape, image_np, frame.header.detections)

            inference_time(value=time_of_inference)
            fps(value=True)
        finally:
            frame.latch.next()
