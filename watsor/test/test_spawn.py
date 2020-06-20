import unittest
import os
from multiprocessing import set_start_method

if __name__ == '__main__':
    # Spawns the processes forcing the system to create copies of stream objects
    # in the scope of a process to validate the data and state are shared properly.
    set_start_method('spawn')

    suite = unittest.defaultTestLoader.discover(start_dir=os.path.dirname(__file__))
    result = unittest.TextTestRunner(failfast=True, descriptions=False, verbosity=2).run(suite)
    if len(result.errors) > 0 or len(result.failures) > 0:
        exit(1)
