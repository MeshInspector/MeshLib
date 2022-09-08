#!/bin/bash
# script to collect compile timing logs

if [[ $# != 3 ]]
then
    echo "invalid argument count"
    exit 1
fi

# param 2 system name (i.e ubuntu20)
system=$1
# param 3 config name (i.e Debug)
config=$2
# param 4 compiler (i.e GCC 10)
compiler=`echo $3 | sed "s/ /_/"`

mkdir -p time_log
for file in `find ./build/$config/*/ -name compile_timings.txt`
do
    echo "collect $file"
    cat $file >> time_log/compile_timings_${system}_${config}_${compiler}.csv
done
for file in `find ./build/$config/*/ -name link_timings.txt`
do
    echo "collect $file"
    cat $file >> time_log/link_timings_${system}_${config}_${compiler}.csv
done
