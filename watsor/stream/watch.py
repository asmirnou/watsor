from threading import Thread
from watsor.stream.spin import Spin


class WatchDog(Spin):
    """Watches the other processes, checking every 10 seconds and restarting them if not alive.
    """

    def __init__(self, name: str, stop_event, log_queue, interval=10, kwargs=None):
        super().__init__(Thread, name, stop_event, log_queue,
                         args=(interval,), kwargs={} if kwargs is None else kwargs)

    __children = []

    @classmethod
    def add_child(cls, child):
        cls.__children.append(child)

    @classmethod
    def remove_child(cls, child):
        cls.__children.remove(child)

    def _run(self, stop_event, log_queue, *args, **kwargs):
        super()._run(stop_event, log_queue, *args, **kwargs)
        self._logger.debug("Started")
        try:
            self._spin(self._watch, stop_event, stop_event, *args, **kwargs)
        except Exception:
            self._logger.exception('Spin failure')
        self._logger.debug("Stopped")

    def _watch(self, stop_event, interval, *args, **kwargs):
        for child in self.__children:
            if child.is_alive():
                self._logger.debug('{} {} ({}) is alive'.format(child.delegate_class_name,
                                                                child.name, child.__class__.__name__))
            elif not child.is_shutdown():
                self._restart(child)

        stop_event.wait(interval)

    def _restart(self, child):
        try:
            self._logger.warning('{} {} ({}) is not alive, restarting...'.format(
                child.delegate_class_name, child.name, child.__class__.__name__))
            child.initialize()
            child.start()
        except AssertionError as e:
            self._logger.error('Failed to restart {} {} ({}). {}.'.format(
                child.delegate_class_name, child.name, child.__class__.__name__, e))
        except Exception:
            self._logger.exception('Failed to restart {} {} ({})'.format(
                child.delegate_class_name, child.name, child.__class__.__name__))
