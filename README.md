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

### Testing the Project

A demonstration script ```scripts/demo.sh``` is avaiable 