from __future__ import print_function

from dataset_utils import DatasetBase
from .. import mygraph
from .. import mygraph_utils as mgutils


class Dataset(DatasetBase):
    @property
    def inittime(self):
        return 0 

    def __init__(self, datafn, localtime, nsteps, stepsize, stepstride, offset=0):
        self.datafn = datafn
        self.__datadir = datafn 

        DatasetBase.__init__(self, datafn, localtime, nsteps, stepsize, stepstride, offset)

    @property
    def name(self):
        return "general"

    # required by Timeline
    def _time2unit(self, tm):
        return int(tm)

    def _unit2time(self, unit):
        return str(unit)

    # required by DyanmicGraph
    def _load_unit_graph(self, tm):
        tm = self._time2unit(tm)
        fn = "{}/{}".format(self.__datadir, tm)
        return mgutils.load_edgelist(fn)

    def _merge_unit_graphs(self, graphs, curstep):
        curunit = self._time2unit(self.step2time(curstep))
        print("merging graph from year {} to {}".format(curunit, curunit + self.stepsize - 1))

        ret = mygraph.Graph(graphs[0].node_type(), graphs[0].weight_type())
        for g in graphs:
            ret.merge(g, free_other=False)

        return ret

    # required by Archivable(Archive and Cache)
    # def _full_archive(self, name=None):
    #     return self.archive(name)

    def archive(self, name=None):
        ar = super(Dataset, self).archive()
        return ar

    def load_archive(self, ar, copy=False, name=None):
        super(Dataset, self).load_archive(ar, copy=copy)


