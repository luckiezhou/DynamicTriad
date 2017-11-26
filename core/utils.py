import sys
import gconfig
from os import path

file_dir = path.dirname(path.abspath(__file__))
build_opts = ['build_ext', '--build-lib', file_dir, '--build-temp', file_dir + '/.cython_build']
_old_argv = sys.argv

try:
    if not gconfig.use_cython:
        raise ImportError("Cython disabled in config")
            
    from Cython.Build import cythonize
    from distutils.core import setup
    sys.argv = sys.argv[:1] + build_opts
    setup(name="utils_cy", ext_modules=cythonize(
        file_dir + "/cython_src/utils_cy.pyx",
        language="c++"
    ))
    sys.argv = _old_argv
    from utils_cy import *
except ImportError as e:
    sys.argv = _old_argv
    print("Cython not avaiable, falling back to python implemented utils")
    print("Err msg: {}".format(e.message))
    from utils_py import *


# TODO: move this into specific implementations later
import multiprocessing as mp
import math
import itertools
import dill
import contextlib
from sklearn.metrics import confusion_matrix


def func_wrapper(args):
    func = dill.loads(args[0])
    return func(mp.current_process()._identity, *args[1:])


# work(process, list_of_samples, reportq): map function, maps a sequence of samples to the sequence of results
# monitor(queue): process monitor function, exit receiveing a StopIteration object
class ParMap(object):
    def __init__(self, work, monitor=None, njobs=mp.cpu_count(), maxtasksperchild=100):
        self.work_func = work
        self.monitor_func = monitor
        self.__njobs = njobs
        self.__mtpc = maxtasksperchild

        self.__pool = None

    def close(self):
        if self.__pool is not None:
            self.__pool.close()
            self.__pool.join()
        self.__pool = None

    def __del__(self):
        self.close()

    @property
    def njobs(self):
        return self.__njobs

    @njobs.setter
    def njobs(self, n):
        self.__njobs = n
        self.close()

    def default_chunk(self, dlen):
        return int(math.ceil(float(dlen) / self.njobs))

    def run(self, data, chunk=None, shuffle=False):
        if chunk is None:
            chunk = self.default_chunk(len(data))

        if shuffle:
            data, order, invorder = shuffle_sample(data, return_order=True)

        slices = slice_sample(data, chunk=chunk)
        res = self.run_slices(slices)

        if shuffle:
            res = apply_order(res, invorder)

        return res

    def run_slices(self, slices):
        mgr = mp.Manager()
        report_queue = mgr.Queue()
        if self.monitor_func is not None:
            monitor = mp.Process(target=self.monitor_func, args=(report_queue,))
            monitor.start()
        else:
            monitor = None

        if self.njobs == 1:
            res = []
            for slc in slices:
                res.append(self.work_func(None, slc, report_queue))
        else:
            dill_work_func = dill.dumps(self.work_func)
            with contextlib.closing(mp.Pool(self.njobs, maxtasksperchild=self.__mtpc)) as pool:
                res = pool.map(func_wrapper, [[dill_work_func, slc, report_queue] for slc in slices])
        res = list(itertools.chain.from_iterable(res))

        report_queue.put(StopIteration())
        if monitor is not None:
            monitor.join()

        return res


def group_by(data, key=lambda x: x):
    ret = []
    key2idx = {}
    for d in data:
        k = key(d)
        idx = key2idx.get(k, None)
        if idx is None:
            idx = key2idx[k] = len(key2idx)
            ret.append([])
        ret[idx].append(d)
    return ret


def confusion_matrix_curve(y):
    y = np.array(y)
    assert np.all((y == 1) + (y == -1)), y
    tp, fp, tn, fn = [0], [0], [np.sum(y == -1)], [np.sum(y == 1)]
    for i in range(len(y)):
        if y[i] == 1:
            tp.append(tp[-1] + 1)
            fp.append(fp[-1])
            tn.append(tn[-1])
            fn.append(fn[-1] - 1)
        elif y[i] == -1:
            tp.append(tp[-1])
            fp.append(fp[-1] + 1)
            tn.append(tn[-1] - 1)
            fn.append(fn[-1])

    return tp, fp, tn, fn


def ks_curve(lb, score, return_score=False):
    # sort by p from largest to smallest
    p, y = zip(*list(sorted(zip(score, lb), key=lambda x: -x[0])))
    tp, fp, tn, fn = confusion_matrix_curve(y)

    tp = np.array(tp, dtype='float32')
    fp = np.array(fp, dtype='float32')
    tn = np.array(tn, dtype='float32')
    fn = np.array(fn, dtype='float32')

    assert np.all(tp + fn == tp[0] + fn[0])
    assert np.all(tn + fp == tn[0] + fp[0])

    pos = tp / (tp + fn)
    neg = 1 - tn / (tn + fp)

    if return_score:
        return pos, neg, p
    else:
        return pos, neg


# determine threshold according to training data
def stdks(train_lb, train_score, test_lb, test_score):
    trainks = ks_curve(train_lb, train_score, return_score=True)
    thidx = np.argmax(trainks[0] - trainks[1])
    sorted_score = trainks[2]
    if thidx == 0:
        th = sorted_score[0] + 1
    elif thidx == len(train_score) + 1:
        th = sorted_score[-1] - 1
    else:
        th = (sorted_score[thidx] + sorted_score[thidx - 1]) / 2

    p = np.sign(test_score - th)
    cmat = confusion_matrix(test_lb, p, labels=(-1, 1)).astype('float32')
    tp, fp, tn, fn = cmat[1, 1], cmat[0, 1], cmat[0, 0], cmat[1, 0]
    pos = tp / (tp + fn)
    neg = 1 - tn / (tn + fp)
    return pos - neg


# determine threshold according to max ks value
def maxks(lb, score):
    pos, neg = ks_curve(lb, score)
    return np.max(pos - neg)

