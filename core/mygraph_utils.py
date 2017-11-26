from __future__ import print_function
import itertools
import mygraph


def type2python(tp):
    if tp == 'string':
        return str
    elif tp in ['short', 'int', 'long', 'long long', 'int16_t', 'int32_t', 'int64_t']:
        return int
    elif tp in ['double', 'float']:
        return float
    else:
        raise TypeError("Unknown type {}".format(tp))


def python2type(tp):
    if tp == int:
        return 'int'  # TODO: long is better, however, mygraph supports only int curretnly
    elif tp == str:
        return 'string'
    elif tp == float:
        return 'float'
    else:
        raise TypeError("Unsupported python type {}".format(tp))


# translate some typical type aliases
def format_type(tp):
    if tp in ['short', 'int16', 'int16_t']:
        return 'int16'
    elif tp in ['int', 'int32', 'int32_t']:
        return 'int32'
    elif tp in ['long', 'long long', 'int64', 'int64_t']:
        return 'int64'
    elif tp in ['float', 'real']:
        return 'float32'
    elif tp in ['double']:
        return 'float64'
    else:
        raise ValueError("Unknown Type {}".format(tp))


# TODO: add undirected mode
def save_graph(g, fn, fmt='adjlist'):
    if fmt == 'adjlist':
        save_adjlist(g, fn)
    # elif fmt == 'edgelist':
    #     save_edgelist(g, fn, weight=weight)
    # elif fmt == 'TNE':
    #     save_TNE(g, fn, weight=weight)
    else:
        raise RuntimeError("Unknown graph format {}".format(fmt))


# graph is directed and weighted by default
def save_adjlist(g, fn):
    fh = open(fn, 'w')
    nodes = list(sorted(g.vertices()))
    for i in nodes:
        nbrs = g.get(i)  # [(nbr, w), ...]
        strnbr = ' '.join([str(e) for e in itertools.chain.from_iterable(nbrs)])
        print("{} {}".format(i, strnbr), file=fh)
    fh.close()


def load_edgelist(fn, node_type='string', weight_type='float'):
    """
    loads only undirected graph, if multiple instances of the same edge is detected,
    there weights are summed up
    :param fn:
    :param node_type:
    :param weight_type:
    :return:
    """
    py_node_type = type2python(node_type)
    py_weight_type = type2python(weight_type)

    g = mygraph.Graph('string', 'float')
    for line in open(fn, 'r'):
        line = line.rstrip('\n').split()
        assert len(line) <= 2, "more than 3 components found in line {}".format(line)
        line[0], line[1] = py_node_type(line[0]), py_node_type(line[1])

        if line[0] == line[1]:
            print("[warning] loopback edge {} ignored".format((line[0], line[1])))
            continue

        if len(line) == 2:
            w = py_weight_type(line[2])
        else:
            w = 1.0

        if g.exists(line[0], line[1]):
            print("[warning] duplicated edge detected: {}".format(line[0], line[1]))

        g.inc_edge(line[0], line[1], w)
        g.inc_edge(line[1], line[0], w)

    return g
