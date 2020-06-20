import unittest
import threading
from watsor.stream.sync import State, StateLatch, CountDownLatch


class TestSync(unittest.TestCase):

    def test_count_down_latch(self):
        def count_down(_latch: CountDownLatch):
            _latch.count_down()

        count = 4
        latch = CountDownLatch(count)
        self.assertFalse(latch.wait(0))

        for i in range(count):
            thread = threading.Thread(target=count_down, args=(latch,))
            thread.start()

        self.assertTrue(latch.wait(5))

    def test_state_transition(self):
        # noinspection PyTypeChecker
        all_states = list(State)

        state = all_states[0]
        for i in range(1, len(all_states)):
            state = State.next(state)
            self.assertEqual(state, all_states[i])

    def test_state_latch_basic(self):
        def next_state(_latch: StateLatch):
            _latch.next()

        # noinspection PyTypeChecker
        all_states = list(State)

        latch = StateLatch()
        self.assertFalse(latch.wait(all_states[-1], 0))

        for i in range(1, len(all_states)):
            thread = threading.Thread(target=next_state, args=(latch,))
            thread.start()

        self.assertTrue(latch.wait(all_states[-1], 5))

    def test_state_latch_count_down(self):
        latch = StateLatch()
        next_state_count_down = 10
        latch.next(next_state_count_down)
        for j in [1, 2]:
            for i in range(j, next_state_count_down):
                old, new = latch.next(i)
                self.assertEqual(new, old)
            old, new = latch.next()
            self.assertNotEqual(new, old)

    def test_state_latch_wait_for(self):
        # noinspection PyTypeChecker
        all_states = list(State)

        latch = StateLatch()
        self.assertFalse(latch.wait(all_states[-1], 0))

        def predicate():
            old, new = latch.next()
            return new == all_states[-1]

        while not latch.wait_for(all_states[-1], predicate, 0.01):
            pass
        self.assertTrue(latch.wait(all_states[-1], 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
