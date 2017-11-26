# distutils: language=c++
from __future__ import print_function

import numpy as np
cimport numpy as np
cimport cython
cimport libc.math as math


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=1] x(int a, int b, int c, object g, np.ndarray[np.float32_t, ndim=2] emb, list nodenames):
    cdef float w1, w2
    w1, w2 = g.edge(nodenames[a], nodenames[c]), g.edge(nodenames[b], nodenames[c])
    return (emb[c] - emb[a]) * w1 + (emb[c] - emb[b]) * w2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float P(int a, int b, int c, object g, np.ndarray[np.float32_t, ndim=2] emb, np.ndarray[np.float32_t, ndim=1] theta,
             list nodenames):
    cdef float power
    power = -(np.dot(theta[:theta.shape[0] - 1], x(a, b, c, g, emb, nodenames)) + theta[theta.shape[0] - 1])
    if power > 100.0:
        return 0.0
    else:
        return 1.0 / (1 + math.exp(power))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list emcoef(list data, np.ndarray[np.float32_t, ndim=2] emb, np.ndarray[np.float32_t, ndim=1] theta,
             object g, list nodenames, dict name2idx, int curstep):
    cdef eps = 1e-6
    cdef int y, k, i, j, lb
    cdef float C, C0, C1
    
    ret = [None] * len(data)

    for didx in range(len(data)):
        y, k, i, j, lb = data[didx][:5]
        wtv = data[didx][5:]
        
        assert y == curstep, "expected step {}, got {}".format(curstep, y)

        if lb == 0:
            C = 1.0
        else:
            C0 = P(i, j, k, g, emb, theta, nodenames)

            # do we need to rewrite this using c++ map?
            inbr = set(list(g.out_neighbours(nodenames[i])))
            jnbr = set(list(g.out_neighbours(nodenames[j])))
            cmnbr = inbr.intersection(jnbr)

            C1 = 1
            for v in cmnbr:
                C1 *= (1 - P(i, j, name2idx[v], g, emb, theta, nodenames))
            C1 = 1.0 - C1

            C = 1 - C0 / (C1 + eps)

            if not np.isfinite(C):
                print(C0, C1, C, [1 - P(i, j, name2idx[v], g, emb, theta, nodenames) for v in cmnbr])
                print(i, j, k)
                print(g.exists(nodenames[i], nodenames[k]),
                      g.exists(nodenames[j], nodenames[k]))
                print([name2idx[n] for n in inbr], [name2idx[n] for n in jnbr])
                assert 0

        ret[didx] = ([y, k, i, j], [C] + wtv)

    return ret

