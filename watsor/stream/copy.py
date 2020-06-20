from io import DEFAULT_BUFFER_SIZE
from watsor.stream.spin import Spin


class Copy(Spin):
    """Copies data from one stream to another.
    """

    def __init__(self, delegate_class, name: str, stop_event, log_queue, src, dst,
                 buffer_size=DEFAULT_BUFFER_SIZE, args=(), kwargs=None):
        super().__init__(delegate_class, name, stop_event, log_queue,
                         args=(src, dst, buffer_size, *args), kwargs={} if kwargs is None else kwargs)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super()._run(stop_event, log_queue, *args, **kwargs)
        try:
            self._spin(self._copy, stop_event, *args, **kwargs)
        except BrokenPipeError:
            pass  # Ignore broken pipe errors, as program exited before all data were written
        except Exception:
            self._logger.exception('Spin failure')
        finally:
            self._close(*args, **kwargs)

    @staticmethod
    def _copy(src, dst, buffer_size, *args, **kwargs):
        buf = src.read(buffer_size)
        if buf:
            dst.write(buf)

    @staticmethod
    def _close(src, dst, *args, **kwargs):
        try:
            dst.close()
        except OSError:
            pass  # Ignore OS errors such as broken pipe, as program most likely exited before all data were written
        finally:
            src.close()
