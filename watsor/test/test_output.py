from unittest import TestCase, installHandler, main
from unittest.mock import MagicMock
from threading import Thread
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue, Event
from time import time
from watsor.stream.share import FrameBuffer, State
from watsor.output.snapshot import Snapshot
from watsor.stream.log import LogHandler
from watsor.stream.work import Payload
from watsor.config.coco import COCO_CLASSES


class MockEffect(object):

    def apply(self, image_in, image_out, shape, header_in, header_out):
        for detection in filter(lambda d: d.label > 0, header_out.detections):
            self.draw_rect(detection.bounding_box.x_min, detection.bounding_box.y_min,
                           detection.bounding_box.x_max, detection.bounding_box.y_max)

    @staticmethod
    def draw_rect(x1, y1, x2, y2):
        pass


class TestOutput(TestCase):

    def setUp(self):
        """Handle Ctrl+C signal to stop tests prematurely.
        """

        installHandler()

    def test_snapshot(self):
        width = 100
        height = 100
        frame_buffer = FrameBuffer(10, width, height)
        frame_queue = Queue(1)

        log_queue = Queue()
        getLogger().addHandler(QueueHandler(log_queue))

        stop_process_event = Event()

        effect = MockEffect()
        effect.draw_rect = MagicMock()

        snapshot = Snapshot("snapshot", stop_process_event, log_queue, frame_queue, frame_buffer,
                            self._create_detect_config(width, height), [effect])
        processes = [snapshot,
                     LogHandler(Thread, "logger", stop_process_event, log_queue, filename=None)]

        for process in processes:
            process.start()

        try:
            frame_index = 0
            frame = frame_buffer.frames[frame_index]

            frame.header.detections[0].label = COCO_CLASSES.index('book')
            frame.header.detections[0].bounding_box.x_min = 1
            frame.header.detections[0].bounding_box.y_min = 2
            frame.header.detections[0].bounding_box.x_max = 3
            frame.header.detections[0].bounding_box.y_max = 4
            frame.header.epoch = time()

            frame.latch.next()
            frame.latch.next()

            payload = Payload(None, frame_index)
            frame_queue.put(payload)

            self.assertTrue(frame.latch.wait_for(State.READY, stop_process_event.is_set, 10))

            with self.assertRaises(AssertionError):
                snapshot.get('person')

            self.assertIsNotNone(snapshot.get('book'))

            effect.draw_rect.assert_called_with(1, 2, 3, 4)
        finally:
            stop_process_event.set()
            for process in processes:
                process.join(30)

    @staticmethod
    def _create_detect_config(width, height):
        return {
            'width': width,
            'height': height,
            'detect': [{'book': {}}]
        }


if __name__ == '__main__':
    main(verbosity=2)
