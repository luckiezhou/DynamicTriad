pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

set -e

MAINPATH=$(dirname $(readlink -f $0))/..
export PYTHONPATH=$MAINPATH
echo setting \$PYTHONPATH to $PYTHONPATH
echo entering directory $MAINPATH
pushd $MAINPATH

if [ -e output ]; then
    if [ -f output/.dynamic_triad ]; then
        rm -rf output
    else
        echo file/directory $MAINPATH/output already exists, please remove it before running the demo 1>&2
        popd
        exit 1
    fi
fi

mkdir -p output
touch output/.dynamic_triad
rm -rf data/academic_toy
mkdir -p data/academic_toy
python scripts/academic2adjlist.py data/academic_toy.pickle data/academic_toy
python . -I 10 -d data/academic_toy -n 15 -K 48 -l 4 -s 2 -o output --beta 1 1 --cachefn /tmp/citation -b 5000
python scripts/stdtests.py -f output -d data/academic_toy.pickle -m 1980 -s 4 -l 2 -n 15 -t all --datasetmod core.dataset.citation --cachefn /tmp/citation

popd
