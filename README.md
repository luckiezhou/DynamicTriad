# DynamicTriad [Under Construction]
This project implements the DynamicTriad algorithm proposed in \[1\], which is a node embedding algorithm for undirected dynamic graphs.

## Building and Testing

This project is implemented primarily in Python 2.7, with some c/c++ extensions written for time efficiency. 

Though the program falls back to pure Python implementation if c/c++ extensions fail to build, we **DISCOURAGE** you from using these code because they might have not been actively maintained and properly tested.

The c/c++ code is **ONLY** compiled and tested with standard GNU gcc/g++ compilers (with c++11 and OpenMP support), and other compilers are explicitly disabled in our build scripts. If you have to use another compiler, modifications on build scripts are required.

### Dependencies

- [Boost.Python](https://www.boost.org/doc/libs/release/libs/python/). Version 1.54.0 has been tested. You can find instructions to install from source [here](http://www.boost.org/doc/libs/1_65_1/libs/python/doc/html/building/installing_boost_python_on_your_.html). 
- [CMake](https://cmake.org).
Version >= 2.8 required. You can find installation instructions [here](https://cmake.org/install/).
- [Eigen 3](https://eigen.tuxfamily.org/).
Version 3.2.8 has been tested, and later versions are expected to be compatible. You can find installation instructions [here](https://eigen.tuxfamily.org/dox/GettingStarted.html).
- [Python 2.7](https://www.python.org).
Version 2.7.13 has been tested. Note that Python development headers are required to build the c/c++ extensions.
- [graph-tool](https://graph-tool.skewed.de).
Version 2.18 has been tested. You can find installation instructions [here](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions).
- [TensorFlow](https://www.tensorflow.org). Version 1.1.0 has been tested. You can find installation instructions [here](https://www.tensorflow.org/install/). Note that the GPU support is **ENCOURAGED** as it greatly boosts training efficiency.
- [Other Python modules](https://pypi.python.org). Some other Python module dependencies are listed in ```requirements.txt```, which can be easily installed with pip:
  ```
  pip install -r requirements.txt
  ```

Although not necessarily mentioned in all the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.

### Building the Project

A building script ```build.sh``` is available in the root directory of this project, simplifying the building process to a single command
```
cd <dynamic_triad_root>; bash build.sh
```
Before running the building commands, you may specify the configuration of some environment variables. You can either use the default value or specify your custom installation paths for certain libraries. For example,
```
PYTHON_LIBRARY? (default: /usr/lib64/libpython2.7.so.1.0, use a space ' ' to leave it empty) 
PYTHON_INCLUDE_DIR? (default: /usr/include/python2.7, use a space ' ' to leave it empty) 
EIGEN3_INCLUDE_DIR? (default: /usr/include, use a space ' ' to leave it empty) 
BOOST_ROOT? (default: , use a space ' ' to leave it empty) ~/boost_1_54_1
```
If everything goes well, the ```build.sh``` script will automate the building process and create all necessary binaries.

Note that the project also contains some Cython modules, however, they will be automatically built as soon as the module is imported in a ready environment.

### Testing the Project

A testing script ```scripts/test.py``` is available, run
```
python scripts/test.py
```
to see if everything is fine with building.

## Usage

Given a sequence of undirected graphs, each for a time step, this program can be used to compute a real-valued vector for each vertex at each time step.

### Data Format

#### Input Format 

The input is expected to be a directory containing ``N`` input files named ``0, 1, 2...``, where `N` is the length of the graph sequence. Each file contains an adjacency list of the corresponding graph, and the adjacency list consists of multiple lines, each in the format:
```
<from_node_name> [<to_node_name1> <weight1> [<to_node_name2> <weight2> ...] ] 
```
where ``x_node_name`` can be any ascii string without white space characters in it, and ``weight`` are float or integer values. The line describes edges from ``from_node_name`` to ``to_node_name1`` and ``to_node_name2`` with weight ``weight1`` and ``weight2`` respectively.

Note that:

- The graph is expected to be undirected, however, it should be presented in a **directed format**. That is, if there is an edge (u, v, w), its reciprocal edge (v, u, w) must also exists in the adjacency list.
- The vertex set should be same for all graphs, if a vertex is missing in a certain graph, simply represent it with an isolated vertex. 
- If a vertex has no outbound vertices, you should write a line with only the ``from_node_name``, instead of ignoring this vertex.
- If the graph is unweighted, place a ``1.0`` for all ``weight`` placeholders, rather than ignoring all weights in the adjacency list.
- Loopback edges (u, u, w) will be ignored when the adjacency list is loaded.

#### Output Format

The program outputs to a directory creating ``N`` files named ``0.out, 1.out, 2.out, ...``, each corresponds to an input file (time step). Each output file contains ``V`` lines, where ``V`` is the number of vertices in each graph. And each line is in format:
```
<node_name> <r1> <r2> ... <rK>
```
where ``<node_name>`` is the name of the vertex defined in the input files, which is followed by ``K`` real values, i.e. the ``K``-length embedding vector for vertex ``<node_name>`` at the corresponding time step. 

### Usage

Assume the input data is available in directory ``<input_dir>`` and the DynamicTriad project root directory is ``<dynamic_triad_root>``, the help information of the embedding program can be obtain by executing command
```
python <dynamic_triad_root> -h
 
usage: <dynamic_triad_root> [-h] [-I NITERS] -d DATAFILE [-b BATCHSIZE] -n NSTEPS
               [-K EMBDIM] [-l STEPSIZE] [-s STEPSTRIDE] -o OUTDIR
               [--cachefn CACHEFN] [--lr LR] [--beta BETA [BETA ...]]
               [--negdup NEGDUP] [--validation VALIDATION]

optional arguments:
  -h, --help            show this help message and exit
  -I NITERS, --niters NITERS
                        number of optimization iterations (default: 10)
  -d DATAFILE, --datafile DATAFILE
                        input directory name (default: None)
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batchsize for training (default: 5000)
  -n NSTEPS, --nsteps NSTEPS
                        number of time steps (default: None)
  -K EMBDIM, --embdim EMBDIM
                        number of embedding dimensions (default: 48)
  -l STEPSIZE, --stepsize STEPSIZE
                        size of of a time steps (default: 1)
  -s STEPSTRIDE, --stepstride STEPSTRIDE
                        interval between two time steps (default: 1)
  -o OUTDIR, --outdir OUTDIR
                        output directory name (default: None)
  --cachefn CACHEFN     prefix for data cache files (default: None)
  --lr LR               initial learning rate (default: 0.1)
  --beta-smooth BETA_SMOOTH
                        coefficients for smooth component (default: 0.1)
  --beta-triad BETA_TRIAD
                        coefficients for triad component (default: 0.1)
  --negdup NEGDUP       neg/pos ratio during sampling (default: 1)
  --validation VALIDATION
                        link_prediction, link_reconstruction, node_classify,
                        node_predict, none (default: link_reconstruction
```

Some of the arguments may require extra explanation:

- ``--beta-smooth/--beta-triad``, two hyper parameters used in the model, see reference [1] for details about the hyper parameters of DynamicTriad. Empirically, the hyper parameters need to be tuned in order to achieve best performance, and the best choice depends on the task and the stability of the target dynamic network.
- ``-l/--stepsize`` and ``-s/--stepstride``, see [Time Model](#time-model) for details.
- ``--cachefn``, sometimes you find that the data preprocessing becomes intolerably time consuming (see [Time Model](#time-model)), and a solution is to specify ``--cachefn`` so that the program creates or uses a cache file of the preprocessed data. The cache file consists of two parts -- a file named ``<--cachefile>.cache`` as well as a file named ``<--cachefile>.cache.args``. If you have changed your configuration for preprocessing, remove ``<--cachefile>.cache.args`` and the cache will be regenerated.
- ``--validation``, the four tasks available for validation is as defined in [1], please refer to the paper for details.

### Demo

We include a toy data set in the ``data`` directory, which is a preprocessed version of ``ACM-Citation-network V8`` from [AMiner](https://www.aminer.cn/citation). We sampled 2000 vertices in the original graph, and the resulting data is archived with Python pickle module to ``data/academic_toy.pickle``.

A demo script is available as ``scripts/demo.sh``, which basically does three things:

- Call ``scripts/academic2adjlist.py`` to convert the toy data to the input format described in [Input](#input-format).
- Call the main script to compute the vertex embeddings and save them to ``output`` directory.
- Call ``scripts/stdtest.py`` to experiment on standard tasks defined in paper [1].

In the demo script, you can find an example for the standard usage of the main script, as well as hints for the usage of the other two scripts if you are interested. 

To execute the demo, run command
```
bash scripts/demo.sh
```

### Time Model

aaa