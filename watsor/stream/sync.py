from multiprocessing import Condition, Queue, Semaphore, RLock, get_context
from multiprocessing.queues import JoinableQueue
from multiprocessing.sharedctypes import Value
from enum import IntEnum
from typing import Dict
from queue import Full
from time import monotonic


class State(IntEnum):
    """Cyclically traverse from one state to another.
    """

    READY = 1
    DETECT = 2
    PUBLISH = 3

    @staticmethod
    def next(s):
        return State.READY if s == State.PUBLISH \
            else State.DETECT if s == State.READY \
            else State.PUBLISH if s == State.DETECT \
            else None


class CountDownLatch(object):
    """A synchronization aid that allows one or more processes to wait until a set of operations
    being performed in other processes completes.

    A CountDownLatch is initialized with a given count. The wait method blocks until the current
    count reaches zero due to invocations of the count_down() method, after which all waiting processes
    are released and any subsequent invocations of wait return immediately.

    The count can be reset, but one need to be 100% sure no other processes are waiting or counting down
    during reset.
    """

    def __init__(self, count: int = 1, lock=None):
        self.__count = Value('i', count, lock=True if lock is None else lock)
        self.__lock = Condition(lock)

    def reset(self, count):
        self.__lock.acquire()
        self.__count.value = count
        self.__lock.release()

    def count_down(self):
        self.__lock.acquire()
        self.__count.value -= 1
        result = self.__count.value
        if self.__count.value <= 0:
            self.__lock.notify_all()
        self.__lock.release()
        return result

    def wait(self, timeout=None):
        self.__lock.acquire()
        result = self.__lock.wait_for(lambda: self.__count.value <= 0, timeout)
        self.__lock.release()
        return result


class StateLatch(object):
    """A synchronization aid that allows one or more processes to wait for state change until a
    set of operations being performed in other processes completes.

    A StateLatch is initialized with a given state. The wait and wait_for methods block until the current
    state changes to the desired state due to invocations of the next() method, after which all waiting
    processes are released and any subsequent invocations of wait or wait_for return immediately.

    While changing state one can set the counter of the next state. The next state won't take effect
    due to invocations of the next() method until the counter reaches zero. This ensures the requested
    number of processes completed their work when state actually changes.

    The counter of next state can be amended without state change using set_next method, but one need to
    be 100% sure no other processes are waiting or changing state at the moment.
    """

    def __init__(self, state=State.READY, lock: RLock = None):
        self.__state = Value('i', state, lock=True if lock is None else lock)
        self.__lock = Condition(lock)
        self.__next_state_count_down = CountDownLatch(0, lock)
        self.__next_state_count_down_max = Value('i', 0, lock=True if lock is None else lock)

    def set_next(self, next_state_count_down):
        self.__lock.acquire()
        self.__next_state_count_down.reset(next_state_count_down)
        self.__next_state_count_down_max.value = 0
        self.__lock.release()

    def next(self, next_state_count_down: int = 0):
        self.__lock.acquire()
        old = State(self.__state.value)
        self.__next_state_count_down_max.value = max(self.__next_state_count_down_max.value, next_state_count_down)
        if self.__next_state_count_down.wait(0) or \
                self.__next_state_count_down.count_down() == 0:
            self.__state.value = State.next(self.__state.value)
            self.__next_state_count_down.reset(self.__next_state_count_down_max.value)
            self.__next_state_count_down_max.value = 0
        new = State(self.__state.value)
        self.__lock.notify_all()
        self.__lock.release()
        return old, new

    def wait(self, state, timeout=None):
        self.__lock.acquire()
        result = self.__lock.wait_for(lambda: self.__state.value == state, timeout)
        self.__lock.release()
        return result

    def wait_for(self, state, predicate, timeout=None):
        """Wait for the desired state or until a condition evaluates to true. predicate must be
        a callable with the result interpreted as a boolean value. A timeout may be provided giving
        the maximum time to wait. While waiting the predicate is being checked every second.

        :param state: state to wait for
        :param predicate: callable function with the result interpreted as a boolean value
        :param timeout: the maximum time to wait
        :return: the last return value of the predicate or False if the method timed out
        """

        self.__lock.acquire()
        try:
            result = self.__state.value == state or predicate()
            if result:
                return result
            end_time = None if timeout is None else monotonic() + timeout
            wait_time = 1
            while not result:
                if end_time is not None:
                    wait_time = min(end_time - monotonic(), 1)
                    if wait_time <= 0:
                        break
                result = self.__lock.wait_for(lambda: self.__state.value == state, wait_time) or predicate()
            return result
        finally:
            self.__lock.release()

    @property
    def state(self):
        return State(self.__state.value)


class BalancedQueue(object):
    """Puts the message in the real queue, but ensures the caller doesn't exceed the limit allowed.
    Throws queue.Full exception when limit is reached. The dictionary of semaphores allows several
    senders to put messages in a single queue, ensuring none of the senders is deprived and they all
    put equal number of messages when receiving party is not capable to process all the messages.
    """

    def __init__(self, delegate: Queue, semaphores: Dict[object, Semaphore], sender=None):
        self.__delegate = delegate
        self.__sender = sender
        self.__semaphores = semaphores

    def put(self, obj, block=True, timeout=None):
        assert self.__sender is not None
        if not self.__semaphores[self.__sender].acquire(block, timeout):
            raise Full
        balanced_put = (self.__sender, obj)
        self.__delegate.put(balanced_put, block, timeout)

    def get(self, block=True, timeout=None):
        balanced_put = self.__delegate.get(block, timeout)
        self.__semaphores[balanced_put[0]].release()
        return balanced_put[1]

    def put_nowait(self, obj):
        return self.put(obj, False)

    def get_nowait(self):
        return self.get(False)

    def qsize(self):
        return self.__delegate.qsize()

    def empty(self):
        return self.__delegate.empty()

    def full(self):
        return self.__delegate.full()

    def close(self):
        self.__delegate.close()

    def join_thread(self):
        self.__delegate.join_thread()

    def cancel_join_thread(self):
        self.__delegate.cancel_join_thread()


class CountableQueue(JoinableQueue):
    """A queue type which automatically calls task_done() allowing to call join() in order to make
    sure all messages are read.
    """

    def __init__(self, *args, **kwargs):
        super(CountableQueue, self).__init__(*args, **kwargs, ctx=get_context())

    def get(self, block=True, timeout=None):
        result = super().get(block, timeout)
        self.task_done()
        return result
