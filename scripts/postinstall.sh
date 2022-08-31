#!/bin/bash

# This script adds MR libs symbolic links to python3
# Expand ld search paths, if `/usr/local/lib` is not added to default

# exit if any command failed
set -eo pipefail

. /etc/lsb-release
UBUNTU_MAJOR_VERSION=${DISTRIB_RELEASE%.*}

PYTHON_VERSION="python3.8"
if [ "$UBUNTU_MAJOR_VERSION" == "22" ]; then
  PYTHON_VERSION="python3.10"
fi

#TODO: handle 'home' python installations (conda, ...)
if [ -d /usr/lib/${PYTHON_VERSION} ]; then
 printf "\r${PYTHON_VERSION} was found                       \n"
 if [ "$EUID" -ne 0 ]; then
  printf "Root access required!\n"
  RUN_AS_ROOT="NO"
 fi
 sudo ln -sf /usr/local/lib/MeshLib/mrmeshpy.so /usr/local/lib/${PYTHON_VERSION}/dist-packages/mrmeshpy.so
 sudo ln -sf /usr/local/lib/MeshLib/mrmeshnumpy.so /usr/local/lib/${PYTHON_VERSION}/dist-packages/mrmeshnumpy.so
 sudo ln -sf /usr/local/lib/MeshLib/mrviewerpy.so /usr/local/lib/${PYTHON_VERSION}/dist-packages/mrviewerpy.so
 printf "Python3 has symlink to MR libs. Run 'sudo ln -sf /usr/local/lib/MeshLib/mr<lib_name>py.so /<pathToPython>/dist-packages/mr<lib_name>py.so' for custom python installations\n"
else
 printf "\r${PYTHON_VERSION} was not found!                  \n"
fi

printf "Updating ldconfig for '/usr/local/lib/MeshLib'\n"
echo "/usr/local/lib/MeshLib" | sudo tee /etc/ld.so.conf.d/local_libs.conf
sudo ldconfig
