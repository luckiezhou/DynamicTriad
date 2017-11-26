from __future__ import print_function
import numpy as np
import codecs
import os
import errno
import re
import subprocess
from StringIO import StringIO
from threading import Thread
from six.moves.queue import Queue, Empty
from six.moves import cPickle
import time
import scipy.sparse as sp
import scipy.spatial.distance as dist
from collections import defaultdict
import itertools
from sklearn.metrics import average_precision_score
import sys
from copy import deepcopy
import random


# support for multiple types of data sources
class IterableFile(object):
    def __init__(self, itr):
        self.__itr = iter(itr)

    def read(self):
        ret = []
        try:
            while True:
                ret.append(next(self.__itr))
        except StopIteration:
            if len(ret) == 0:
                raise EOFError

        return '\n'.join(ret)

    def readline(self):
        try:
            return next(self.__itr)
        except StopIteration:
            raise EOFError

    def close(self):
        pass

    def __iter__(self):
        return self.__itr


def open_datasrc(ds, encoding=None):
    if isinstance(ds, str):
        try:
            if os.path.isfile(ds):
                if encoding is not None:
                    return codecs.open(ds, 'r', encoding=encoding)
                else:
                    return open(ds, 'r')
        except TypeError:
            pass
        
        return StringIO(ds)
    elif hasattr(ds, 'read') and hasattr(ds, 'readline'):
        return ds
    elif hasattr(ds, '__iter__'):
        return IterableFile(ds)
    else:
        raise NotImplementedError


def save_csr(obj, fn):
    np.savez(fn, data=obj.data, indices=obj.indices, indptr=obj.indptr, shape=obj.shape)


def load_csr(fn):
    npzfiles = np.load(open_datasrc(fn))
    return sp.csr_matrix((npzfiles['data'], npzfiles['indices'], npzfiles['indptr']), shape=npzfiles['shape'])


# save and load a list of objects, handles sparse matrices
def save_arr(obj, fn):
    objcpy = obj
    for i, o in enumerate(obj):
        if isinstance(o, sp.csr_matrix):
            objcpy[i] = {'data': o.data, 'indices': o.indices, 'indptr': o.indptr, 'shape': o.shape}
    cPickle.dump(objcpy, open(fn, 'w'))


def load_arr(fn):
    arr = cPickle.load(open_datasrc(fn))
    for i, o in enumerate(arr):
        if isinstance(o, dict) and 'indptr' in o:
            arr[i] = sp.csr_matrix((o['data'], o['indices'], o['indptr']), shape=o['shape'])
    return arr 


# this method stacks and saves a list of features (for each sample), 
# and it handles dense and sparse matrices automatically
def savefeatlist(fn, arr, dtype=None):
    if len(arr) > 0 and isinstance(arr[0], sp.csr_matrix):
        arr = sp.vstack(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        print("type {} array with shape {}".format(type(arr), arr.get_shape()))
        save_csr(arr, fn)
    else:
        if isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        print("type {} array with shape {}".format(type(arr), arr.shape))
        np.save(fn, arr)


def save_dict(arr, fn, encoding="utf-8"):
    fh = codecs.open(fn, "w", encoding=encoding)
    for i, s in enumerate(arr):
        print(i, s, file=fh)
    fh.close()


def load_dict(fn, encoding="utf-8"):
    fh = open_datasrc(fn, encoding=encoding)
    arr = []
    for l in fh.read().rstrip('\n').split('\n'):
        spl = l.split(' ')
        assert len(spl) <= 2, u"{}: {}".format(l, len(spl))
        if len(spl) == 1:
            arr.append('')
        else:
            arr.append(spl[1])
    fh.close()
    return arr


def load_freq(fn, encoding="utf-8"):
    fh = open_datasrc(fn, encoding=encoding)
    arr = []
    for l in fh.read().rstrip('\n').split('\n'):
        spl = l.split(' ')
        assert len(spl) == 2, u"{}: {}".format(l, len(spl))
        arr.append(spl[0])
    fh.close()
    return arr


def save_freq(freq, fn, encoding='utf-8'):
    if isinstance(freq, (list, tuple)):
        freq = {e: 1 for e in freq}

    fh = codecs.open(fn, "w", encoding=encoding)
    for k in sorted(freq.keys()):
        print(str(k) + ' ' + str(freq[k]), file=fh)
    fh.close()


def load_map(fn, skip_lines=0, key_type=str):
    fh = open_datasrc(fn)
    txt = fh.read().rstrip('\n').split('\n')
    ret = {}
    for line in txt[skip_lines:]:
        kv = line.split(None, 1)
        if len(kv) != 2:
            continue
        ret[key_type(kv[0])] = kv[1]
    return ret


def load_by_suffix(fn):
    suffix = fn[fn.rfind('.') + 1:]
    if suffix == 'csr':
        return load_csr(fn)
    elif suffix == 'npy':
        return np.load(fn)
    elif suffix == 'pickle':
        return cPickle.load(open(fn, 'r'))
    elif suffix == 'dict':
        return load_dict(fn)
    elif suffix == 'freq':
        return load_freq(fn)
    else:
        raise RuntimeError("unrecognized suffix in {}".format(fn))


def ls(dn, filt, basename=False):
    files = [f for f in os.listdir(dn) if re.match(filt, f)]
    files = list(sorted(files))
    if basename:
        if not dn.endswith('/'):
            dn += '/'
        files = [dn + f for f in files] 
    return files


def getlines(fn):
    return int(subprocess.check_output(['sh', '-c', 'sed -n \'=\' ' + fn + ' | wc -l']))


# get lines counts of patterned files in a directory
# def readsubdir(dn, pat):
#     ret = os.popen('perl readsubdir.pl {} {}'.format(dn, pat)).read()
#     return ret.rstrip('\n').split(' ')


class LineRetriever(object):
    def __init__(self, datafn):
        self.proc = subprocess.Popen('./getline ' + datafn, shell=True,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.outq = Queue()
        self.errq = Queue()
        self.outread = Thread(target=self.__enqueue, args=(self.proc.stdout, self.outq))
        self.errread = Thread(target=self.__enqueue, args=(self.proc.stderr, self.errq))
        self.outread.daemon = True
        self.errread.daemon = True
        self.outread.start()
        self.errread.start()

        # test
        self.proc.stdin.write('0\n')
        begint = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(self.proc.returncode, self.__getbuffered(self.errq))
            try:
                self.outq.get_nowait()
                break
            except Empty:
                pass
            if time.time() - begint > 30:
                raise RuntimeError("Time out waiting response from line indexer")
        
        for line in self.__getbuffered(self.errq):
            print(line.rstrip('\n'))

    @staticmethod
    def __getbuffered(q):
        ret = []
        try:
            while True:
                ret.append(q.get_nowait())
        except Empty:
            pass
        return ret
    
    @staticmethod
    def __enqueue(fd, q):
        for line in iter(fd.readline, ''):
            q.put(line)
        fd.close()

    def getline(self, lineno):
        self.proc.stdin.write(str(lineno) + '\n')

        out = None
        begint = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(self.proc.returncode, self.__getbuffered(self.errq))
            try:
                out = self.outq.get_nowait()
                break
            except Empty:
                pass
            if time.time() - begint > 30:
                raise RuntimeError("Time out waiting response from line indexer")

        assert len(self.__getbuffered(self.outq)) == 0
        return out

#    def __destroy__(self):
#        self.proc.terminate()


class LineRetrieverInMem(object):
    # an inmemory verison of LineRetriever with the sampe interface

    def __init__(self, datafn):
        self.data = open(datafn, 'r').read().rstrip('\n').split('\n')

    def getline(self, lineno):
        return self.data[lineno]


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def slice_sample(sample, chunk=None, nslice=None):
    # type: (iterable, int, int) -> iterable
    slices = []
    if chunk is None:
        chunk = int(len(sample) / nslice)
    else:
        if nslice is not None:
            raise RuntimeError("chunk ({}) and slice ({}) should not be specified simultaneously".format(chunk, nslice))

    curstart = 0
    while True:
        if curstart >= len(sample):
            break
        slices.append(sample[curstart:min(curstart + chunk, len(sample))])
        curstart += chunk

    return slices


def islice_sample(sample, chunk=None, nslice=None):
    if chunk is None:
        chunk = int(len(sample) / nslice)
    else:
        if nslice is not None:
            raise RuntimeError("chunk ({}) and slice ({}) should not be specified simultaneously".format(chunk, nslice))

    curstart = 0
    while True:
        if curstart >= len(sample):
            break
        yield sample[curstart:min(curstart + chunk, len(sample))]
        curstart += chunk


def shuffle_sample(sample, return_order=False):
    # type: (iterable) -> tuple
    order = np.random.permutation(np.arange(len(sample)))
    invorder = np.zeros((len(sample), ), dtype='int32')
    invorder[order] = np.arange(len(sample))
    
    if return_order:
        return apply_order(sample, order), order, invorder
    else:
        return apply_order(sample, order)


def apply_order(sample, order):
    return [sample[o] for o in order]


def base_n(num, b, numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (base_n(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])


def unitvec(sz, pos):
    ret = np.zeros((sz, ))
    if pos < 0:
        return ret
    ret[pos] = 1
    return ret


def mycos(v1, v2):
    if all([e == 0 for e in v1]) or all([e == 0 for e in v2]):
        return 0

    return dist.cosine(v1, v2)


def basename(s):
    return s[s.rfind('/') + 1:]


class LenGen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def ilen(gen, length):
    return LenGen(gen, length)


def project(emb, normvec):
    return emb - np.dot(emb, normvec)[:, None] * normvec[None, :]


def mean_average_precision(gt, pred):
    val_score = 0
    for curp, curlb in itertools.izip(pred, gt):
        val_score += average_precision_score(curlb, curp)
    val_score /= len(gt)
    return val_score


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def format_dir(path):
    if not path.endswith('/'):
        path += '/'
    if path.startswith('~/'):
        path = os.getenv('HOME') + '/' + path[2:]
    return path


class ConstantDict(object):
    def __init__(self, value):
        self.__value = value

    def __getitem__(self, key):
        return self.__value


# archive protocol
class Archivable(object):
    def archive(self, name=None):
        return {}

    def load_archive(self, ar, copy=False, name=None):
        pass


class OffsetList(Archivable):
    def __init__(self, offset, length, datasrc, copy=True, managed=None):
        self.offset = offset
        self.length = length
        self.__managed = managed
        if hasattr(datasrc, '__getitem__'):
            if not copy:
                self.__items = datasrc
                if self.__managed is None:
                    self.__managed = False
            else:
                self.__items = deepcopy(datasrc)
                if self.__managed is None:
                    self.__managed = True
            self.__factory = lambda x: self.__items[x - self.offset]
        else:
            self.__items = [None] * self.length
            self.__factory = datasrc
            if self.__managed is None:
                self.__managed = True 
        self.__iter = 0
        self.__accessed = {}

    def __len__(self):
        return self.length
    
    def __normalize_slice(self, slc):
        start, stop, step = slc.start, slc.stop, slc.step
        if slc.start is None:
            start = self.offset
        if slc.stop is None:
            stop = self.offset + self.length
        if slc.step is None:
            step = 1
        start = self.__normalize_neg_index(start)
        stop = self.__normalize_neg_index(stop)
        return slice(start, stop, step)

    def __normalize_neg_index(self, idx):
        if idx < 0:
            return idx + self.offset + self.length
        else:
            return idx

    def __setitem__(self, key, item):
        if isinstance(key, slice):
            slc = self.__normalize_slice(key)
            rg = range(slc.start, slc.stop, slc.step)

            if len(rg) != len(item):
                raise ValueError("Trying to set {} items with {} value".format(len(rg), len(item)))
            
            if slc.start < self.offset or slc.stop > self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(slc, self.offset, self.offset + self.length))
            for step, itm in zip(rg, item):
                self._store_item(step, itm)
        else:
            key = self.__normalize_neg_index(key)
            if key < self.offset or key >= self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(key, self.offset, self.offset + self.length))
            self._store_item(key, item)

    def __getitem__(self, key):
        if key == 'any':
            return self._load_item(self.offset)

        if isinstance(key, slice):
            slc = self.__normalize_slice(key)
            rg = range(slc.start, slc.stop, slc.step)
            
            if slc.start < self.offset or slc.stop > self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(slc, self.offset, self.offset + self.length))
            items = []
            for step in rg:
                items.append(self._load_item(step))
            return items
        else:
            key = self.__normalize_neg_index(key)
            if key < self.offset or key >= self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(key, self.offset, self.offset + self.length))
            return self._load_item(key)
    
    def __array__(self):
        # np.array fails when len(self) == 1, I have no idea why this happens but have to specify array interface manually
        return np.asarray(list(self))

    def __iter__(self):
        self.__iter = self.offset
        return self

    # for py3 compatibility
    def __next__(self):
        return self.next()

    def append(self, item):
        if not self.__managed:
            raise RuntimeError("Cannot append to unmanaged OffsetList")
        self.length += 1
        self.__items.append(item)

    def extend(self, lst):
        if not self.__managed:
            raise RuntimeError("Cannot extend an unmanaged OffsetList")
        self.length += len(lst)
        self.__items.extend(lst)

    def next(self):
        if self.__iter >= self.offset + self.length:
            raise StopIteration
        ret = self._load_item(self.__iter)
        self.__iter += 1
        return ret

    def _load_item(self, step):
        # do some caching
        if self.__accessed.get(step, None) is None:
            self.__items[step - self.offset] = self.__factory(step)
            self.__accessed[step] = True
        return self.__items[step - self.offset]

    def _store_item(self, step, itm):
        self.__items[step - self.offset] = itm
        self.__accessed[step] = True

    def archive(self, name=None):
        if name is None:
            prefix = 'OffsetList'
        else:
            prefix = '{}_OffsetList'.format(name)

        ar = super(OffsetList, self).archive(name)
        ar['{}_offset'.format(prefix)] = self.offset
        ar['{}_length'.format(prefix)] = self.length
        ar['{}_data'.format(prefix)] = list(self)
        return ar

    def load_archive(self, ar, copy=False, name=None):
        if name is None:
            prefix = 'OffsetList'
        else:
            prefix = '{}_OffsetList'.format(name)

        self.__init__(ar['{}_offset'.format(prefix)], ar['{}_length'.format(prefix)], ar['{}_data'.format(prefix)], copy=copy)


# target: output target
# max_num: maximum number of logs to print, -1 for Infinity
# nodup: ignore duplicated records
# delayed: output in conclude method instead of log method
class Logger(object):
    def __init__(self, target=sys.stdout, max_num=None, nodup=False, delayed=False):
        self.target = target
        self.max_num = max_num
        self.cnt = 0
        self.nodup = nodup
        self.delayed = delayed

        self.keyset = set()
        self.logs = []

    def __log(self, msg):
        if self.delayed:
            self.logs.append(msg)
        else:
            print(msg, file=self.target)

    def log(self, msg, key=None):
        if self.nodup:
            assert key is not None, "A key should be provided in nodup mode"
            if key in self.keyset:
                return
            self.keyset.add(key)

        if self.max_num is not None and self.cnt < self.max_num:
            self.__log(msg)
        self.cnt += 1

    def conclude(self, msg=None):
        if msg is None:
            msg = '{} logs in total'

        if self.delayed:
            print('\n'.join(self.logs), file=self.target)
        if self.max_num is not None and self.cnt > self.max_num:
            print('... and {} more logs'.format(self.cnt - self.max_num), file=self.target)
        if self.cnt > 0:
            print(msg.format(self.cnt))


# returns random value between 0 and 1
# this method is not c-implemented, it is called crandom for compatibility
def crandint(ub):
    return random.randint(ub - 1)
