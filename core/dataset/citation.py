from __future__ import print_function

# import graphtool_utils as gtutils
import numpy as np
import re
from six.moves import cPickle, reduce
from collections import Counter, defaultdict
import random
from dataset_utils import DatasetBase
import core.gconfig as gconf
from core import utils, mygraph


class Dataset(DatasetBase):
    @property
    def inittime(self):
        return self.__data['args']['minyear']

    def __init__(self, datafn, localyear, nsteps, stepsize=5, stepstride=1, offset=0):
        self.datafn = datafn
        self.__data = cPickle.load(open(self.datafn, 'r'))

        DatasetBase.__init__(self, datafn, localyear, nsteps, stepsize, stepstride, offset)

        self.__vertex_raw_labels_cache = None

    @property
    def name(self):
        return "citation"

    # required by Timeline
    def _time2unit(self, tm):
        return int(tm)

    def _unit2time(self, unit):
        return str(unit)

    # required by DyanmicGraph
    def _load_unit_graph(self, tm):
        year = self._time2unit(tm)
        # return gtutils.load_graph(self.__data['graphs'][year],
        #                           fmt='mygraph', convert_to='undirected')
        return self.__data['graphs'][year]

    def _merge_unit_graphs(self, graphs, curstep):
        # nodecnt = graphs['any'].num_vertices()
        # rawname = graphs['any'].vertex_properties['name']
        # curunit = self._time2unit(self.step2time(curstep))
        #
        # g = gt.Graph(directed=False)
        # g.add_vertex(nodecnt)
        # w = g.new_ep('float')
        # g.edge_properties['weight'] = w
        # name = g.new_vp('int')
        # name.a = rawname.a
        # g.vertex_properties['name'] = name
        #
        # print("merging graph from year {} to {}".format(curunit, curunit + self.stepsize - 1))
        # edge_cache = defaultdict(lambda: 0)
        # for i in range(len(graphs)):
        #     wi = graphs[i].edge_properties['weight']
        #     for e in graphs[i].edges():
        #         isrc, itgt = int(e.source()), int(e.target())
        #         isrc, itgt = min(isrc, itgt), max(isrc, itgt)
        #         edge_cache[(isrc, itgt)] += wi[e]
        # edgearr = np.zeros((len(edge_cache), 3))
        # for i, (k, v) in enumerate(edge_cache.items()):
        #     edgearr[i] = [k[0], k[1], v]
        # g.add_edge_list(edgearr, eprops=[w])

        curunit = self._time2unit(self.step2time(curstep))
        print("merging graph from year {} to {}".format(curunit, curunit + self.stepsize - 1))

        ret = mygraph.Graph(graphs[0].node_type(), graphs[0].weight_type())
        for g in graphs:
            ret.merge(g, free_other=False)

        return ret
        # g = gtutils.merge_graph(graphs, graphs['any'].vp['name'], [g.ep['weight'] for g in graphs])
        # return g

    # required by Archivable(Archive and Cache)
    def _full_archive(self, name=None):
        self.__vertex_raw_labels()  # evaluate lazy operations
        return self.archive(name)

    def archive(self, name=None):
        if name is None:
            prefix = 'Dataset'
        else:
            prefix = '{}_Dataset'.format(name)

        ar = super(Dataset, self).archive()
        ar['{}_cache'.format(prefix)] = [self.__vertex_raw_labels_cache]
        return ar

    def load_archive(self, ar, copy=False, name=None):
        if name is None:
            prefix = 'Dataset'
        else:
            prefix = '{}_Dataset'.format(name)

        super(Dataset, self).load_archive(ar, copy=copy)
        self.__vertex_raw_labels_cache, = ar['{}_cache'.format(prefix)]
        if copy:
            self.__vertex_raw_labels_cache = self.__vertex_raw_labels_cache.copy()

    @property
    def manual_features(self):
        raise NotImplementedError()

    @property
    def data(self):
        return self.__data

    @staticmethod
    def __label_vertices(feats, featnames, confdata):
        labels = []
        for f in feats:
            cur = [0] * len(confdata)
            for idx in np.nonzero(f)[0]:
                curconfidx = None
                for k, v in confdata.items():
                    if re.match(v[1], featnames[idx]) or re.match(v[2], featnames[idx]):
                        if curconfidx is None:
                            curconfidx = v[0]
                        else:
                            print("[Warning]: {} satisfies both patterns {} and {}".format(featnames[idx], curconfidx,
                                                                                           v[0]))
                if curconfidx is not None:
                    cur[curconfidx] += f[idx]
            if np.max(cur) <= 0:
                labels.append(-1)
            else:
                labels.append(np.argmax(cur))
        print("label distribution: {}".format(Counter(labels)))
        return np.array(labels)

    def __vertex_raw_labels(self, return_name=False):
        raw_names = {-1: 'Unknown', 0: 'Architecture', 1: 'Computer Network', 2: 'Computer Security',
                     3: 'Data Mining', 4: 'Theory', 5: 'Graphics'}

        if self.__vertex_raw_labels_cache is not None:
            if return_name:
                return self.__vertex_raw_labels_cache, raw_names
            else:
                return self.__vertex_raw_labels_cache

        # These are conferences that are to merged
        confdata = [['ASPLOS|Architectural Support for Programming Languages and Operating Systems',
                     'FAST|Conference on File and Storage Technologies',
                     'HPCA|High-Performance Computer Architecture',
                     'ISCA|Symposium on Computer Architecture',
                     'MICRO|MICRO',
                     'USENIX ATC|USENIX Annul Technical Conference',
                     'PPoPP|Principles and Practice of Parallel Programming'],
                    ['MOBICOM|Mobile Computing and Networking Transactions on Networking',
                     'SIGCOMM|applications, technologies, architectures, and protocols for computer communication',
                     'INFOCOM|Computer Communications'],
                    ['CCS|Computer and Communications Security',
                     'NDSS|Network and Distributed System Security',
                     # 'CRYPTO|International Cryptology Conference',
                     # 'EUROCRYPT|European Cryptology Conference',
                     'S\&P|Symposium on Security and Privacy',
                     'USENIX Security|Usenix Security Symposium'],
                    ['SIGMOD|Conference on Management of Data',
                     'SIGKDD|Knowledge Discovery and Data Mining',
                     'SIGIR|Research on Development in Information Retrieval',
                     'VLDB|Very Large Data Bases',
                     'ICDE|Data Engineering'],
                    ['STOC|ACM Symposium on Theory of Computing',
                     'FOCS|Symposium on Foundations of Computer Science',
                     'LICS|Symposium on Logic in Computer Science',
                     'CAV|Computer Aided Verification'],
                    [  # 'ACM MM|Multimedia',
                        'SIGGRAPH|SIGGRAPH Annual Conference',
                        'IEEE VIS|Visualization Conference',
                        'VR|Virtual Reality'],
                    # ['AAAI|AAAI Conference on Artificial Intelligence',
                    #  'CVPR|Computer Vision and Pattern Recognition',
                    #  'ICCV|International Conference on Computer Vision',
                    #  'ICML|International Conference on Machine Learning',
                    #  'IJCAI|International Joint Conference on Artificial Intelligence',
                    #  'NIPS|Annual Conference on Neural Information Processing Systems',
                    #  'ACL|Annual Meeting of the Association for Computational Linguistics']
                    ]
        confdata = {n: i for i, arr in enumerate(confdata) for n in arr}
        for k in confdata.keys():
            sname, lname = k.split('|')
            confdata[k] = [confdata[k], re.compile(sname), re.compile(lname, re.I)]
        conffeat = [self.__data['conf_feat'][y] for y in range(self.localunit, self.localunit + self.nunits)]
        # conffeat = utils.load_arr(self.datadir + '/conf_feat.pickle')
        # conffeat, conffeatargs = conffeat[:-1], conffeat[-1]

        conffeat_names = self.__data['conf_names']
        # conffeat_names = utils.load_dict(self.datadir + '/conf_names.dict')
        confmap = self.__data['confmap']
        # confmap = utils.load_map(self.datadir + "/conference.map")
        conffeat_names = [confmap[c] for c in conffeat_names]

        # we use theory conferences because it is more independent
        rawlb = []
        for i in range(self.nsteps):
            startunit = self._time2unit(self.step2time(i + self.localstep))
            endunit = startunit + self.stepsize
            relstartunit, relendunit = startunit - self.localunit, endunit - self.localunit
            print("generating samples for years from {} to {}, i.e. featidx from {} to {}".
                  format(startunit, endunit - 1, relstartunit, relendunit - 1))
            curconffeat = reduce(lambda x, y: x + y, conffeat[relstartunit:relendunit],
                                 np.zeros(conffeat[relstartunit].shape, dtype=conffeat[relstartunit].dtype)).A
            rawlb.append(self.__label_vertices(curconffeat, conffeat_names, confdata))

            print("{}/{} positive samples at step {}".format(np.sum(rawlb[-1] == 1), len(rawlb[-1]), i + self.localstep))
        rawlb = np.vstack(rawlb)
        self.__vertex_raw_labels_cache = rawlb
        if return_name:
            return self.__vertex_raw_labels_cache, raw_names
        else:
            return self.__vertex_raw_labels_cache

    # unlike classification_samples, this method returns labels for all nodes from time begin to end
    # with pos labeled as 1, neg labeled as -1 and unknown labeled as 0
    def vertex_labels(self, target=4, return_name=False):
        rawlb, raw_names = self.__vertex_raw_labels(return_name=True)
        if target == 'raw':
            lb = rawlb
            label_names = raw_names
        else:
            lb = rawlb.copy()
            # TODO: make sure the order of lb (i.e. order of feat) is 0:nnodes
            def mapper(x):
                if x == target:
                    return 1
                elif x == -1:
                    return 0
                else:
                    return -1
            lb = np.vectorize(mapper)(lb)
            #lb[lb != target] = -1
            #lb[lb == target] = 1
            label_names = {-1: 'Others', 1: raw_names[target], 0: 'Unknown'}

        assert lb.shape == (len(self.gtgraphs), self.gtgraphs['any'].num_vertices()), \
            "{}, ({}, {})".format(lb.shape, len(self.gtgraphs), self.gtgraphs['any'].num_vertices())
        
        if return_name:
            return utils.OffsetList(self.localstep, len(lb), lb, copy=False), label_names
        else:
            return utils.OffsetList(self.localstep, len(lb), lb, copy=False)

    # TODO: consider moving following code into another module
    # def vertex_classify_samples(self, begin, end):
    #     lb = np.array(self.vertex_labels()[begin:end])
    #     lb = np.reshape(lb, (lb.size, ), order='C')
    #
    #     samp = []
    #     nodecnt = self.gtgraphs['any'].num_vertices()
    #     for i in range(begin, end):
    #         samp.append(np.transpose(np.vstack((i * np.ones((nodecnt, ), dtype='int32'),
    #                                             np.arange(nodecnt, dtype='int32')))))
    #     samp = np.concatenate(samp, axis=0)
    #
    #     assert samp.shape == (lb.shape[0], 2), "{}, {}".format(samp.shape, (lb.shape[0], 2))
    #     return samp, lb
    #
    # # sample_method in 'negsamp', 'changed'
    # #   negsamp: all samples with proper negative sampling
    # #   changed: all changed edges
    # # negrange: used only in negsamp mode, cache to select negative samples from
    # # TODO: rewrite this with mygraph
    # def link_predict_samples(self, begin, end, sample_method='negsamp', negrange=None, negdup=1):
    #     if sample_method == 'negsamp':
    #         pos = []
    #         for i, g in enumerate(self.gtgraphs[begin:end]):
    #             for e in g.edges():
    #                 if int(e.source()) > int(e.target()):  # because our graph is undirected
    #                     # check symmetric
    #                     names = g.vertex_properties['name']
    #                     assert g.edge(e.target(), e.source()), "{}: {} {}".format(i + begin, names[e.source()],
    #                                                                               names[e.target()])
    #                     assert g.edge_properties['weight'][e] == g.edge_properties['weight'][g.edge(e.target(), e.source())]
    #                     continue
    #                 pos.append([i + begin, int(e.source()), int(e.target())])
    #         pos = np.vstack(pos).astype('int32')
    #
    #         def make_cache(x):
    #             cache = []
    #             g = self.gtgraphs[x]
    #             all_nodes = set(range(g.num_vertices()))
    #             for j in range(g.num_vertices()):
    #                 # TODO: note here we assume undirected graph
    #                 cache.append(list(all_nodes - set([int(v) for v in g.vertex(j).all_neighbours()])))
    #             return cache
    #         if negrange is None:
    #             negrange = utils.KeyDefaultDict(make_cache)
    #
    #         neg = []
    #         for i in range(negdup):
    #             for p in pos:
    #                 tm, src, tgt = p
    #                 if random.randint(0, 1) == 0:  # replace source
    #                     cur_range = negrange[tm][tgt]
    #                     new_src = cur_range[random.randint(0, len(cur_range) - 1)]
    #                     neg.append([tm, new_src, tgt])
    #                 else:  # replace target
    #                     cur_range = negrange[tm][src]
    #                     new_tgt = cur_range[random.randint(0, len(cur_range) - 1)]
    #                     neg.append([tm, src, new_tgt])
    #         neg = np.vstack(neg).astype('int32')
    #
    #         lbs = np.concatenate((np.ones(len(pos)), -np.ones(len(neg))))
    #         return np.concatenate((pos, neg), axis=0), lbs
    #     elif sample_method == 'changed':
    #         if end - begin < 2:
    #             raise RuntimeError("there must be at least 2 graphs in 'changed' sample method")
    #
    #         samp, lbs = [], []
    #         for i in range(begin, end - 1):
    #             prevg, curg = self.gtgraphs[i], self.gtgraphs[i + 1]
    #
    #             def edge_set(g):
    #                 ret = set()
    #                 for e in g.edges():
    #                     s, t = int(e.source()), int(e.target())
    #                     if s > t:
    #                         s, t = t, s
    #                     ret.add((s, t))
    #                 return ret
    #
    #             cure = edge_set(curg)
    #             preve = edge_set(prevg)
    #
    #             for s, t in cure - preve:
    #                 # i + 1 because i enumerates all prev graphs
    #                 samp.append([i + 1, s, t])
    #                 lbs.append(1)
    #             for s, t in preve - cure:
    #                 samp.append([i + 1, s, t])
    #                 lbs.append(-1)
    #
    #             if gconf.debug:
    #                 # only check in debug mode because it is time consuming to call g.edge
    #                 for i in range(len(samp)):
    #                     if lbs[i] == 1:
    #                         assert self.gtgraphs[samp[i][0]].edge(samp[i][1], samp[i][2]) is not None
    #                     else:
    #                         assert self.gtgraphs[samp[i][0]].edge(samp[i][1], samp[i][2]) is None
    #
    #         samp = np.array(samp)
    #         lbs = np.array(lbs)
    #
    #         return samp, lbs
    #     else:
    #         raise NotImplementedError()

