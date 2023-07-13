__author__ = 'arthur'
import sys
from queue import Empty
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np

import logging
#logging.basicConfig(level=logging.DEBUG, filename='SDE.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('SDE.log')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)

def print_debug(*msg):
    if '-debug' in sys.argv:
        str = ','.join([repr(x) for x in msg])
        logger.debug(str)
    #logger.handlers[0].flush()
    #sys.stdout.flush()

#mp.set_start_method('fork')

# import cProfile

def _queue_iter(queue):
    while not queue.empty():
        yield queue.get()


def _serial_map(func, args_list):
    results = []
    for args in args_list:
        if isinstance(args, dict):
            res = func(**args)
        elif isinstance(args, str):
            res = func(args)
        elif hasattr(args, '__iter__'):
            res = func(*args)
        else:
            res = func(args)
        results.append(res)
    return results


def _parallel_run(func, inputs, outputs):
    while not inputs.empty():
        try:
            i, args = inputs.get_nowait()
            if isinstance(args, dict):
                res = func(**args)
            elif isinstance(args, str):
                res = func(args)
            elif hasattr(args, '__iter__'):
                res = func(*args)
            else:
                res = func(args)
            outputs.put((i, res))
        except Empty:
            continue


def _run_jobs(work, args_list, num_cores):
    jobs = []
    for rank in range(min(len(args_list), num_cores)):
        job = mp.Process(target=work, name="worker_%s" % rank, args=(rank,))
        jobs.append(job)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()


def parallel_map(func, args_list, num_cores=16):
    if not hasattr(args_list, '__len__'): args_list = list(args_list)
    if num_cores <= 0: return _serial_map(func, args_list)
    inputs = mp.Queue()
    outputs = mp.Queue()

    for i, args in enumerate(args_list):
        inputs.put((i, args))

    def work(rank):
        print_debug("starting core %d" % rank)
        _parallel_run(func, inputs, outputs)

    _run_jobs(work, args_list, num_cores)
    return [value for _, value in sorted(_queue_iter(outputs))]


class ConstSharedArray:
    def __init__(self, array):
        self.shm = SharedMemory(create=True, size=array.nbytes)
        self.array = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)
        self.array[:] = array
        self.array.flags.writeable = False
        self.main = True

    def __getitem__(self, key):
        return self.array[key]

    def tofile(self, fname):
        self.array.tofile(fname)

    def __setstate__(self, state):
        name, shape, dtype = state
        self.shm = SharedMemory(name=name, create=False)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        self.array.flags.writeable = False
        self.main = False

    def __getstate__(self):
        return (self.shm.name, self.array.shape, self.array.dtype)

    def __del__(self):
        self.shm.close()
        if self.main:
            self.shm.unlink()

    def __len__(self):
        return len(self.array)


class WritableSharedArray:
    def __init__(self, array):
        self.shm = SharedMemory(create=True, size=array.nbytes)
        self.array = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)
        self.array[:] = array
        self.array.flags.writeable = True
        self.main = True

    def __getitem__(self, key):
        return self.array[key]

    def tofile(self, fname):
        self.array.tofile(fname)
    def __setitem__(self, key, value):
        self.array.__setitem__(key, value)

    def clear(self, key):
        if isinstance(self.array[key],np.ndarray):
            self.array[key].fill(0)
        else:
            self.array[key] =0
    def __setstate__(self, state):
        name, shape, dtype = state
        self.shm = SharedMemory(name=name, create=False)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        self.array.flags.writeable = True
        self.main = False

    def __getstate__(self):
        return (self.shm.name, self.array.shape, self.array.dtype)

    def __del__(self):
        self.shm.close()
        if self.main:
            self.shm.unlink()

    def __len__(self):
        return len(self.array)
