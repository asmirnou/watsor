from unittest import TestCase, installHandler, main
from threading import Thread
from itertools import cycle
from time import time, sleep
from logging import getLogger, ERROR
from logging.handlers import QueueHandler
from multiprocessing import Process, Event, Queue
from watsor.stream.log import LogHandler
from watsor.stream.share import FrameBuffer, NonSharedFramesPerSecond, FramesPerSecond
from watsor.output.video import VisualEffects
from watsor.stream.sync import State, CountDownLatch
from watsor.test.numpy_stream import NumpyRead, NumpyWork, simple_math
from watsor.test.detect_stream import Artist, ShapeDetector, ShapeCounter
from watsor.output.copy import CopyImageEffect, CopyHeaderEffect
from watsor.output.draw import DrawEffect


class TestShare(TestCase):

    def setUp(self):
        """Handle Ctrl+C signal to stop tests prematurely.
        """

        installHandler()

    def test_simple_math(self):
        self.assertEqual(36, simple_math('*', 12, 3))
        self.assertEqual(40, simple_math('/', 80, 2))
        self.assertEqual(56, simple_math('+', 50, 6))
        self.assertEqual(15, simple_math('-', 20, 5))

    def test_frame_buffer(self):
        frame_buffer = FrameBuffer(10, 1, 1, 1)

        length = len(frame_buffer.frames)
        frame_cycle = cycle(range(length))
        for i in range(-1, length * 3):
            _, frame_index = frame_buffer.select_next_ready(i)
            self.assertEqual(next(frame_cycle), frame_index)

    def test_frame_buffer_expire(self):
        getLogger(FrameBuffer.__name__).setLevel(ERROR)

        frame_buffer = FrameBuffer(1, 1, 1, 1)
        for frame in frame_buffer.frames:
            frame.header.epoch = time()
            frame.latch.next()
        frame, frame_index = frame_buffer.select_next_ready()
        self.assertIsNone(frame)

        for frame in frame_buffer.frames:
            frame.header.epoch -= 60
        frame, frame_index = frame_buffer.select_next_ready()
        self.assertIsNotNone(frame)

    def test_numpy_stream(self):
        """Tests shared memory usage across processes. One process fills a frame buffer
        with random data, while another performs simple math operations on the given random image,
        comparing with predicted result.
        """

        frame_buffer = FrameBuffer(5, 10, 10, 1, 'd')

        frame_queue = Queue()

        log_queue = Queue()
        getLogger().addHandler(QueueHandler(log_queue))

        stop_process_event = Event()

        log_handler = LogHandler(Process, "logger", stop_process_event, log_queue, filename=None)
        reader = NumpyRead(Process, "reader", stop_process_event, log_queue, frame_queue, frame_buffer)
        worker = NumpyWork(Process, "worker", stop_process_event, log_queue, frame_queue, frame_buffer)

        log_handler.start()
        reader.start()
        worker.start()

        try:
            # Wait till the last frame in the buffer is used.
            self.assertTrue(frame_buffer.frames[-1].latch.wait(State.PUBLISH, 5))
        finally:
            stop_process_event.set()
            reader.join(30)
            worker.join(30)
            log_handler.join(30)

        self.assertEqual(len(frame_buffer.frames), frame_buffer.status[State.PUBLISH])
        self.assertEqual(first=len(frame_buffer.frames),
                         second=worker.matches,
                         msg="Not all frames were processed")

    def test_subscribe(self):
        """Tests the coherence among the processes sharing the same frame buffer.
        The queues tying the processes are of limited size (1) and the workers are a bit slower,
        than the reader. The reader has to drop frames if any of the workers is busy handling previous
        frame. The frame state latch must get back to READY state, if neither the main worker nor subscriber
        can pick up the next frame. This ensures the buffer will never overflow.
        """

        width = 500
        height = 500
        frame_buffer_in = FrameBuffer(10, width, height)
        frame_buffer_out = FrameBuffer(10, width, height)

        frame_queue = Queue(1)
        subscriber_queue = Queue(1)
        subscriber_queue1 = Queue(1)
        subscriber_queue2 = Queue(1)
        subscriber_queue3 = Queue(1)

        log_queue = Queue()
        getLogger().addHandler(QueueHandler(log_queue))

        stop_process_event = Event()

        latch = CountDownLatch(100)

        effects = [CopyHeaderEffect(), CopyImageEffect(), DrawEffect()]

        artist = Artist("artist", stop_process_event, log_queue, frame_queue, frame_buffer_in)
        conductor = VisualEffects("conductor", stop_process_event, log_queue, subscriber_queue,
                                  frame_buffer_in, frame_buffer_out, effects)

        processes = [artist, conductor,
                     LogHandler(Thread, "logger", stop_process_event, log_queue, filename=None),
                     ShapeDetector(Process, "detector", stop_process_event, log_queue, frame_queue, frame_buffer_in),
                     ShapeCounter(Thread, "counter1", stop_process_event, log_queue, subscriber_queue1,
                                  frame_buffer_out, latch),
                     ShapeCounter(Thread, "counter2", stop_process_event, log_queue, subscriber_queue2,
                                  frame_buffer_out, latch),
                     ShapeCounter(Thread, "counter3", stop_process_event, log_queue, subscriber_queue3,
                                  frame_buffer_out, latch)]

        artist.subscribe(subscriber_queue)
        conductor.subscribe(subscriber_queue1)
        conductor.subscribe(subscriber_queue2)
        conductor.subscribe(subscriber_queue3)

        for process in processes:
            process.start()

        try:
            self.assertTrue(latch.wait(15))
        finally:
            stop_process_event.set()
            for process in processes:
                process.join(30)

            conductor.unsubscribe(subscriber_queue1)
            conductor.unsubscribe(subscriber_queue2)
            conductor.unsubscribe(subscriber_queue3)
            artist.unsubscribe(subscriber_queue)

    def test_nonshared_frames_per_second(self):
        fps = NonSharedFramesPerSecond()

        thread = Thread(target=self.some_work, args=(fps, 10, 0.05))
        thread.start()
        thread.join(30)

        self.assertAlmostEqual(20, fps(), delta=3, msg="FPS measured is wrong")

    def test_frames_per_second(self):
        fps = FramesPerSecond()

        process = Process(target=self.some_work, args=(fps, 10, 0.05))
        process.start()
        process.join(30)

        self.assertAlmostEqual(20, fps(), delta=3, msg="FPS measured is wrong")

    @staticmethod
    def some_work(fps, r, s):
        for i in range(r):
            fps(value=True)
            sleep(s)


if __name__ == '__main__':
    main(verbosity=2)
