#!/bin/bash

# This script adds MR libs symbolic links to python3
# Expand ld search paths, if `/usr/local/lib` is not added to default

# exit if any command failed
set -eo pipefail

#TODO: handle 'home' python installations (conda, ...)
if [ -d /usr/lib/python3.9 ]; then
 printf "\rPython3 was found                       \n"
 if [ "$EUID" -ne 0 ]; then
  printf "Root access required!\n"
  RUN_AS_ROOT="NO"
 fi
 sudo ln -sf /usr/local/lib/MeshLib/mrmeshpy.so /usr/lib/python3.9/dist-packages/mrmeshpy.so
 sudo ln -sf /usr/local/lib/MeshLib/mrmeshnumpy.so /usr/lib/python3.9/dist-packages/mrmeshnumpy.so
 sudo ln -sf /usr/local/lib/MeshLib/mrealgorithmspy.so /usr/lib/python3.9/dist-packages/mrealgorithmspy.so
 printf "Python3 has symlink to MR libs. Run 'sudo ln -sf /usr/local/lib/MeshLib/mr<lib_name>py.so /<pathToPython>/dist-packages/mr<lib_name>py.so' for custom python installations\n"
fi

printf "Updating ldconfig for '/usr/local/lib/MeshLib'\n"
echo "/usr/local/lib/MeshLib" | sudo tee /etc/ld.so.conf.d/local_libs.conf
sudo ldconfig
