#!/bin/bash

# for github runners only!

PRECOMPILED_LIB_PATH="./../saved_libs/meshlib_lib"
DO_RECOMPILE=0

while getopts ":p:rh" opt;
do
  case $opt in
    p)
      PRECOMPILED_LIB_PATH=$OPTARG
      ;;
    r)
      echo "recompile was triggered"
      DO_RECOMPILE=1
      ;;
    h)
      echo " -p PATH_TO - set path to precompiled libs"
      echo " -r - to recompile thirdparty"
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

PRECOMPILED_LIB_PATH=$(realpath -m "${PRECOMPILED_LIB_PATH}")

rm -rf ./lib

if [ ! -d "${PRECOMPILED_LIB_PATH}" ] || [ $DO_RECOMPILE == 1 ]; then
  rm -rf ${PRECOMPILED_LIB_PATH}
  mkdir -p ${PRECOMPILED_LIB_PATH}
  ./scripts/build_thirdparty.sh
  cp -r "./lib" "${PRECOMPILED_LIB_PATH}"
else
  ln -s ${PRECOMPILED_LIB_PATH} "./lib"
  echo "Link to ${PRECOMPILED_LIB_PATH} created."
fi
exit 0