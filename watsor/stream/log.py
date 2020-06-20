import traceback
import logging
import logging.handlers
from sys import stdout, stderr
from watsor.stream.work import Work


class LogHandler(Work):
    """Performs the logging in a console and a file, receiving the messages from multiple processes
    through the queue. While the normal messages are being written to the console, teh errors are being
    reported to stderr. The log file is being rolled over after the current file reaches a certain size.
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, filename,
                 max_bytes=10 * 1024 * 1024, backup_count=5, kwargs=None):
        super().__init__(delegate_class, name, stop_event, log_queue, log_queue,
                         args=(filename, max_bytes, backup_count),
                         kwargs={} if kwargs is None else kwargs)

    def _config_logger(self, log_queue, *args, **kwargs):
        self._config_log_handers(*args, **kwargs)

    def _config_log_handers(self, _, filename, max_bytes, backup_count, *args, **kwargs):
        self._logger = logging.Logger(self.__class__.__name__)

        console_stdout = logging.StreamHandler(stdout)
        formatter = logging.Formatter('%(threadName)-16s %(name)-24s %(levelname)-8s: %(message)s')
        console_stdout.addFilter(lambda record: record.levelno < logging.ERROR)
        console_stdout.setFormatter(formatter)

        console_stderr = logging.StreamHandler(stderr)
        console_stderr.setLevel(logging.ERROR)
        console_stderr.setFormatter(formatter)

        self._logger.addHandler(console_stdout)
        self._logger.addHandler(console_stderr)

        if filename is not None:
            file = logging.handlers.RotatingFileHandler(filename, 'a', max_bytes, backup_count)
            formatter = logging.Formatter('%(asctime)s %(threadName)-16s %(name)-24s %(levelname)-8s: %(message)s')
            file.setFormatter(formatter)
            if 'DEBUG' == kwargs.get('log_level'):
                self._logger.debug("Log is being written to %s", filename)
            self._logger.addHandler(file)

    def _next_frame(self, record, *args, **kwargs):
        try:
            self._logger.handle(record)
        except Exception:
            traceback.print_exc()
