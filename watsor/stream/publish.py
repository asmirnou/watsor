from queue import Full
from collections import Counter


class Publish(object):
    """This class keeps the list of subscribers and publishes a message only to
    those who can consume it. Thus if a subscriber is busy processing previous
    frame, it merely misses the current one. This logic ensures a slow worker
    won't make others to wait.
    """

    def __init__(self, lock):
        self._subscribers = Counter()
        self._subscribers_lock = lock

    def subscribe(self, a_queue):
        self._subscribers_lock.acquire()
        try:
            self._subscribers[a_queue] += 1
        finally:
            self._subscribers_lock.release()

    def unsubscribe(self, a_queue):
        self._subscribers_lock.acquire()
        try:
            self._subscribers[a_queue] -= 1
            if self._subscribers[a_queue] == 0:
                del self._subscribers[a_queue]
        finally:
            self._subscribers_lock.release()

    def _publish(self, payload):
        self._subscribers_lock.acquire()
        try:
            count = 0
            for a_queue in self._subscribers.keys():
                try:
                    a_queue.put_nowait(payload)
                    count += 1
                except Full:
                    pass
            return count
        finally:
            self._subscribers_lock.release()
