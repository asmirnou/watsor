from signal import signal, SIGINT
from logging import Logger, getLogger, INFO
from multiprocessing import Queue, Event, get_start_method
from threading import current_thread, main_thread
from logging.handlers import QueueHandler


class Spin(object):
    """Base class for any repeatable action. Encapsulates a Thread or a Process instance.
    Configures logging system for an instance of a derived class sending logs through the
    queue to another process.
    """

    def __init__(self, delegate_class, name: str, stop_event: Event, log_queue: Queue, args=(), kwargs=None):
        self._logger = None
        self.__delegate_class = delegate_class
        self.__name = name
        self.__stop_event = stop_event
        self.__log_queue = log_queue
        self.__args = args
        self.__kwargs = {} if kwargs is None else kwargs
        self.__delegate = None
        self.initialize()

    def initialize(self):
        """This is method is public to let watchdog restart delegate process if not alive. The spin
        instance keeps all arguments and starts the process as from the beginning.
        """

        assert self.__delegate is None or not self.__delegate.is_alive(), \
            "{} has not terminated yet".format(self.delegate_class_name)

        # noinspection PyAttributeOutsideInit
        self.__delegate = self.__delegate_class(name=self.__name, target=self._run,
                                                args=(self.__stop_event, self.__log_queue, *self.__args),
                                                kwargs=self.__kwargs)

    @property
    def delegate_class_name(self):
        return self.__delegate.__class__.__name__

    @property
    def name(self):
        return self.__name

    @staticmethod
    def _spin(action, stop_event, *args, **kwargs):
        while not stop_event.is_set():
            action(*args, **kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        current_thread().name = self.__name

        if current_thread() is main_thread() and \
                get_start_method() == 'spawn':
            signal(SIGINT, self._signal_handler)

        self._config_logger(log_queue, *args, **kwargs)

    def _config_logger(self, log_queue, *args, **kwargs):
        if self._logger is not None:
            return

        if current_thread() is main_thread():
            # If we're here, then we're in a new process and need to send all logs to the queue,
            self._logger = Logger(self.__class__.__name__)
            self._logger.addHandler(QueueHandler(log_queue))
        else:
            # Otherwise it is assumed that the main thread's already configured root logger
            # to interact with the queue.
            self._logger = getLogger(self.__class__.__name__)

        self._logger.setLevel(kwargs.get('log_level', INFO))

    def start(self):
        self.__delegate.start()

    def terminate(self):
        self.__stop_event.set()

    def join(self, timeout=None):
        self.__delegate.join(timeout)

    def is_alive(self):
        return self.__delegate.is_alive()

    def is_shutdown(self):
        return self.__stop_event.is_set()

    def _signal_handler(*args):
        pass


class Stub(object):
    """Pretends to be a Thread or a Process, when a derived from Spin class
    is being executed in already running Thread, such as a thread of HTTP server.
    """

    def __init__(self, name: str, target, args=(), kwargs=None):
        self.__name = name
        self.__target = target
        self.__args = args
        self.__kwargs = {} if kwargs is None else kwargs

    @property
    def name(self):
        return self.__name

    def start(self):
        self.__target(*self.__args, **self.__kwargs)

    @staticmethod
    def join(timeout=None):
        pass

    @staticmethod
    def is_alive():
        return True
