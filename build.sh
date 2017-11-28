pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

set -e

filedir=$(dirname $(readlink -f $0))
echo entering $filedir
pushd $filedir

ask() {
    local ret
    echo -n "$1 (default: $2, use a space ' ' to leave it empty) " 1>&2
    read ret
    if [ -z "$ret" ]; then
        ret=$2
    elif [ "$ret" == " " ]; then
        ret=""
    fi
    echo $ret
}

echo "You may need to specify some environments before building"

pylib=$(python -c "from distutils.sysconfig import get_config_var; print('{}/{}'.format(get_config_var('LIBDIR'), get_config_var('INSTSONAME')))")
pylib=$(ask "PYTHON_LIBRARY?" $pylib)
export PYTHON_LIBRARY=$pylib

pyinc=$(python -c "from distutils.sysconfig import get_config_var; print(get_config_var('INCLUDEPY'))")
pyinc=$(ask "PYTHON_INCLUDE_DIR?" $pyinc)
export PYTHON_INCLUDE_DIR=$pyinc

eigeninc=$(ask "EIGEN3_INCLUDE_DIR" /usr/include)
export EIGEN2_INCLUDE_DIR=$eigeninc

boostroot=$(ask "BOOST_ROOT" "")
export BOOST_ROOT=$boostroot

echo building mygraph module ...
rm -r core/mygraph-build
mkdir -p core/mygraph-build
pushd core/mygraph-build
cmake ../graph
make && make install
popd

popd
