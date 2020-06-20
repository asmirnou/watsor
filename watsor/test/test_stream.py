import unittest
import multiprocessing as mp
import threading
import time
import itertools
from logging import getLogger
from logging.handlers import QueueHandler
from collections import defaultdict
from statistics import stdev
from watsor.stream.log import LogHandler
from watsor.stream.sync import BalancedQueue
from watsor.test.dummy_stream import DummyRead, DummyWork


class TestStream(unittest.TestCase):
    """Tests various configurations of streaming
    """

    def setUp(self):
        """Handle Ctrl+C signal to stop tests prematurely.
        """

        unittest.installHandler()

        self.stop_main_event = threading.Event()
        self.stop_process_event = mp.Event()

    def test_lag_absence(self):
        """Absence of lag during streaming, when the readers are much faster than workers
        Test configuration is set up so that the reader is several times faster then the worker.
        If buffering is configured, a chronic lag interferes the viewing of the processed stream in
        real time. The reader should drop the frames exceeded the buffer, but the buffer also should
        be small enough (up to 1 frame) to get rid of noticeable delay. If buffering is configured,
        the maximum lag between frames will match the reader's time: the frames are close to each
        other, but they are all in queue, which lags the viewing. Here we ensure the buffer is short
        and the maximum delay equals to worker's time, which means they are as up to date as possible.
        """

        # Prepare configuration
        distribution_queue = mp.Queue()
        log_handler, readers, workers, queues, semaphores = self.create_config(
            mp.Process, distribution_queue,
            read_config=[30],
            work_config=[10])

        # Run test for a second, then collect the distribution
        timeout = 1.05
        self.go(log_handler + readers + workers, timeout)

        distribution = queue_to_dict(distribution_queue)
        read_distr = select_and_strip(distribution, 'reader')
        work_distr = select_and_strip(distribution, 'worker')

        # Ensure the amount of data read and processed are the same
        self.assertEqual(sum(read_distr.values()), sum(work_distr.values()),
                         "The number of read and processed frames differ")

        # Ensure the maximum lag matches worker's time, not reader's time
        self.assertAlmostEqual(workers[0].max_lag, 1 / workers[0].fps, places=1,
                               msg="Maximum lag between frames differs from worker's time")

    def test_distribution_uniformity(self):
        """Uniform distribution, when capacity of readers well exceeds the workers
        Test configuration is set up so that total FPS capacity of the readers
        well exceeds FPS capacity of the workers. In this case the readers will compete
        for pushing their chunks of data to a slow worker. They won't be able to push all the
        data, but we need to ensure none of the readers is deprived and they all distributed
        almost equal number of frames. At that the workers will process the streams at their
        full capacity, which is configured different. We need to ensure there were no skews
        in workload.
        """

        # Prepare configuration
        distribution_queue = mp.Queue()
        log_handler, readers, workers, queues, semaphores = self.create_config(
            mp.Process, distribution_queue,
            read_config=[100, 300, 100, 300],
            work_config=[10, 50, 200])

        # Run test for a second, then collect the distribution
        timeout = 1.05
        self.go(log_handler + readers + workers, timeout)

        distribution = queue_to_dict(distribution_queue)
        read_distr = select_and_strip(distribution, 'reader')
        work_distr = select_and_strip(distribution, 'worker')

        # Ensure the amount of data read and processed are the same
        self.assertEqual(sum(read_distr.values()), sum(work_distr.values()),
                         "The number of read and processed frames differ")

        # Ensure the readers distribute chunks of data uniformly
        self.assertLess(stdev(read_distr.values()), 15,
                        "The readers did not distribute frames uniformly: {}"
                        .format(list(read_distr.values())))

        # Ensure the workload across the workers is distributed as configured
        for (i, j) in itertools.permutations(range(len(workers)), 2):
            self.assertAlmostEqual(first=workers[i].fps / workers[j].fps,
                                   second=work_distr[str(i)] / work_distr[str(j)],
                                   delta=5.0,
                                   msg="The workload per two workers ({} and {}) did not fit "
                                       "their configuration: planned [{}/{}], but was [{}/{}]"
                                   .format(i, j, workers[i].fps, workers[j].fps,
                                           work_distr[str(i)], work_distr[str(j)]))

    def test_idyll(self):
        """Idyll, when the workers can consume all data produced by the readers
        Test configuration is set up so that total FPS capacity of the workers
        well exceeds FPS capacity of the readers. In this case the readers will produce
        data at full capacity as the workers are able to process all frames.
        """

        # Prepare configuration
        distribution_queue = mp.Queue()
        log_handler, readers, workers, queues, semaphores = self.create_config(
            threading.Thread, distribution_queue,
            read_config=[30, 10, 30, 10],
            work_config=[160, 50])

        # Run test for a second, then collect the distribution
        timeout = 1.05
        timeout = self.go(log_handler + readers + workers, timeout)

        distribution = queue_to_dict(distribution_queue)
        read_distr = select_and_strip(distribution, 'reader')
        work_distr = select_and_strip(distribution, 'worker')

        # Ensure the amount of data read and processed are the same
        self.assertEqual(sum(read_distr.values()), sum(work_distr.values()),
                         "The number of read and processed frames differ")

        # Ensure each reader was used at full capacity
        for i in range(len(read_distr)):
            self.assertAlmostEqual(first=readers[i].fps,
                                   second=read_distr[str(i)] / timeout,
                                   delta=2.5,
                                   msg="The reader {} was not used at full capacity: planned {}, but was {}"
                                   .format(i, readers[i].fps, read_distr[str(i)] / timeout))

        # Ensure each reader's produced as much data as was configured, comparing with others
        for (i, j) in itertools.permutations(range(len(read_distr)), 2):
            self.assertAlmostEqual(first=readers[i].fps / readers[j].fps,
                                   second=read_distr[str(i)] / read_distr[str(j)],
                                   delta=5.0,
                                   msg="Two readers ({} and {}) produced diverse amount "
                                       "of data than was configured: planned [{}/{}], but was [{}/{}]"
                                   .format(i, j, readers[i].fps, readers[j].fps,
                                           read_distr[str(i)], read_distr[str(j)]))

    def create_config(self, delegate_class, distribution_queue, read_config, work_config):
        """Prepares configuration of a test.

        :param delegate_class: Process or Thread
        :param distribution_queue: queue to return distribution after test finishes
        :param read_config: FPS configuration of the readers
        :param work_config: FPS configuration of the workers
        :return: arrays of readers and workers
        """

        log_queue = mp.Queue()
        getLogger().addHandler(QueueHandler(log_queue))
        log_handler = LogHandler(delegate_class, "LogHandler", self.stop_process_event, log_queue, filename=None)

        frame_queue = mp.Queue()
        all_semaphores = {}

        readers = []
        for pos in range(len(read_config)):
            reader_name = "reader {}".format(pos)
            reader_queue_semaphore = mp.BoundedSemaphore(1)
            all_semaphores[reader_name] = reader_queue_semaphore
            decoder_queue = BalancedQueue(frame_queue, {reader_name: reader_queue_semaphore}, reader_name)

            readers.append(DummyRead(delegate_class, reader_name, self.stop_process_event, log_queue,
                                     decoder_queue, read_config[pos]))

        workers = []
        for pos in range(len(work_config)):
            workers.append(DummyWork(delegate_class, "worker {}".format(pos), self.stop_process_event, log_queue,
                                     BalancedQueue(frame_queue, all_semaphores),
                                     distribution_queue, work_config[pos]))

        return [log_handler], readers, workers, \
               [log_queue, frame_queue], all_semaphores

    def go(self, processes, timeout):
        """Starts all processes, waits till timeout exceeded,
        then terminates the processes.

        :param processes: list of processes to start
        :param timeout: timeout in seconds
        :return: time elapsed while waiting
        """

        for process in processes:
            process.start()

        timeout = self.wait(processes, timeout)

        self.stop_process_event.set()
        for process in processes:
            process.join(30)

        return timeout

    def wait(self, processes, timeout):
        """While waiting till timeout exceeded, checks each process is alive,
        stopping if not.

        :param processes: list of processes to check
        :param timeout: timeout in seconds
        :return: time elapsed while waiting
        """

        start_time = time.time()
        duration = 0
        while not self.stop_main_event.wait(timeout / 10) \
                and duration < timeout:
            if not all(map(lambda process: process.is_alive(), processes)):
                break
            duration = time.time() - start_time
        return time.time() - start_time


def queue_to_dict(queue):
    """Read all tuples from the queue, summing up the values for similar keys.

    :param queue: the queue to get tuples
    :return: dictionary with distribution data
    """

    result = defaultdict(int)
    while not queue.empty():
        (key, count) = queue.get()
        result[key] += count
    return result


def select_and_strip(dictionary, prefix):
    """Selects the keys from the dictionary by prefix,
    returning the filters dictionary with prefix stripped.

    :param dictionary: dictionary to filter
    :param prefix: prefix to match the keys
    :return: filtered dictionary
    """

    return {k[len(prefix):].strip(): v for k, v in dictionary.items() if k.startswith(prefix)}


if __name__ == '__main__':
    unittest.main(verbosity=2)
