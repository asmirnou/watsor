import unittest
from logging import getLogger, ERROR
from logging.handlers import QueueHandler
from threading import Thread
from multiprocessing import Event, Queue
from watsor.stream.log import LogHandler
from watsor.stream.watch import WatchDog
from watsor.stream.sync import CountDownLatch
from watsor.test.dummy_stream import Stumble


class TestWatch(unittest.TestCase):

    def setUp(self):
        """Handle Ctrl+C signal to stop tests prematurely.
        """

        unittest.installHandler()

    def test_watch_dog(self):
        """Runs the watchdog and a child thread that stumbles all the time,
        forcing the watchdog to restart it. Counts the number of restarts during timeout.
        """

        log_queue = Queue()
        stop_process_event = Event()

        getLogger().addHandler(QueueHandler(log_queue))

        latch = CountDownLatch(3)

        log_handler = LogHandler(Thread, "logger", stop_process_event, log_queue, filename=None)
        watch_dog = WatchDog("watchdog", stop_process_event, log_queue, 0.1, kwargs={'log_level': ERROR})
        stumble = Stumble(Thread, "tumbler", stop_process_event, log_queue, 0.1, latch)

        log_handler.start()
        watch_dog.start()
        stumble.start()

        watch_dog.add_child(log_handler)
        watch_dog.add_child(stumble)

        self.assertTrue(latch.wait(5))
        stop_process_event.set()

        watch_dog.remove_child(log_handler)
        watch_dog.remove_child(stumble)

        stumble.join(30)
        watch_dog.join(30)
        log_handler.join(30)


if __name__ == '__main__':
    unittest.main(verbosity=2)
