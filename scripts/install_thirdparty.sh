#!/bin/bash

# This script copy built thirlparty to `/usr/local/lib`
# Generally, better choise is to install deb package

# exit if any command failed
set -eo pipefail

if [ "$MR_STATE" != "DOCKER_BUILD" ]; then
 read -t 10 -p "It is strongly recommended to use 'apt install ./meshlib*.deb' instead! Press (y) in 10 seconds to continue (y/N)" -rsn 1
 echo;
 if ! [[ $REPLY =~ ^[Yy]$ ]]; then
  exit 0
 fi
else
  MR_THIRDPARTY_DIR="."
fi

#install MR requirements
MR_LIB_DIR="./lib/"
MR_INSTALL_THIRDPARTY_DIR="/usr/local/lib/"
MR_INSTALL_FONTS_DIR="/usr/local/share/fonts/"
MR_INSTALL_INCLUDE_DIR="/usr/local/include/"

if [ -z "$MR_THIRDPARTY_DIR" ]; then
  MR_THIRDPARTY_DIR="./thirdparty"
fi

if [ "$EUID" -ne 0 ]; then
 printf "Root access required!\n"
 RUN_AS_ROOT="NO"
fi

cd ${MR_LIB_DIR}
# copy libs
find . -name \*.so* -exec sudo cp -R --preserve=links -p {} ${MR_INSTALL_THIRDPARTY_DIR} \;
#copy fonts
find . -name \*.ttf -exec sudo cp -p {} ${MR_INSTALL_FONTS_DIR} \;
find . -name \*.otf -exec sudo cp -p {} ${MR_INSTALL_FONTS_DIR} \;
cd -

# headers copy
cd ${MR_THIRDPARTY_DIR}
find . -name '*.h' -type f -exec sudo cp -fr --parents \{\} ${MR_INSTALL_INCLUDE_DIR} \;
cd -
cd ${MR_LIB_DIR}
find . -name '*.h' -type f -exec sudo cp -fr --parents \{\} ${MR_INSTALL_INCLUDE_DIR} \;
cd -

if [ "${RUN_AS_ROOT}" = "NO" ]; then
 sudo -k
fi
printf "Thirdparty installation done!\n"
