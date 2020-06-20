import os
from unittest import TestCase, installHandler, main
from test.support import EnvironmentVarGuard
from threading import Thread
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Process, Event, Queue
from watsor.stream.log import LogHandler
from watsor.stream.share import FrameBuffer
from watsor.stream.sync import CountDownLatch
from watsor.detection.detector import create_object_detectors
from watsor.filter.sieve import DetectionSieve
from watsor.filter.track import TrackFilter
from watsor.filter.confidence import ConfidenceFilter
from watsor.stream.share import RateLimiter
from watsor.config.coco import get_coco_class
from watsor.test.detect_stream import Artist, ShapeCounter


class TestDetect(TestCase):

    def setUp(self):
        if not os.path.exists(self._get_model_path()):
            self.skipTest('Test skipped due to the absence of shape object model')

        installHandler()

    def test_shape_detection(self):
        """Tests TensorFlow object detection using the trained model of simple geometric shapes.
        The detector recognises shapes drawn on a frame image, the sieve filters those having confidence
        above 50%, lastly shapes are counted signalling to end the test.
        """

        with EnvironmentVarGuard() as env:
            env.set("TF_CPP_MIN_LOG_LEVEL", "3")
            env.set("CORAL_VISIBLE_DEVICES", "")
            env.set("CUDA_VISIBLE_DEVICES", "")

            frame_buffer = FrameBuffer(10, 100, 100)

            frame_queue = Queue(1)
            subscriber_queue = Queue(1)
            detection_sieve_queue = Queue(1)

            log_queue = Queue()
            getLogger().addHandler(QueueHandler(log_queue))

            stop_process_event = Event()

            latch = CountDownLatch(100)

            artist = Artist("artist", stop_process_event, log_queue, frame_queue, frame_buffer)

            detection_sieve = DetectionSieve("sieve", stop_process_event, log_queue,
                                             detection_sieve_queue, frame_buffer,
                                             self._create_filters(), RateLimiter())

            processes = [artist, detection_sieve,
                         LogHandler(Thread, "logger", stop_process_event, log_queue, filename=None),
                         ShapeCounter(Thread, "counter", stop_process_event, log_queue, subscriber_queue,
                                      frame_buffer, latch)]

            processes += create_object_detectors(Process, stop_process_event, log_queue, frame_queue,
                                                 {artist.name: frame_buffer}, self._get_model_path())

            artist.subscribe(detection_sieve_queue)
            detection_sieve.subscribe(subscriber_queue)

            for process in processes:
                process.start()

            try:
                self.assertTrue(latch.wait(15))
            finally:
                stop_process_event.set()
                for process in processes:
                    process.join(30)

    @staticmethod
    def _create_filters():
        return [TrackFilter([ConfidenceFilter({
            'detect': [{
                get_coco_class(1).label: {  # triangle
                    'confidence': 50
                }}, {
                get_coco_class(2).label: {  # ellipse
                    'confidence': 50
                }}, {
                get_coco_class(3).label: {  # rectangle
                    'confidence': 50
                }}]})], 1, 1)]

    @staticmethod
    def _get_model_path():
        return os.path.join(os.path.dirname(__file__), 'model')


if __name__ == '__main__':
    main(verbosity=2)
