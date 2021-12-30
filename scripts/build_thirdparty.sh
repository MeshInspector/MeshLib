#!/bin/bash

# This script builds thirdparty
# Output libraries are stored in `./lib` directory

# exit if any command failed
set -eo pipefail

dt=$(date '+%d-%m-%Y_%H:%M:%S');
logfile="`pwd`/install_thirdparty_${dt}.log"
printf "Thirdparty build script started.\nYou could find output in ${logfile}\n"

#update submodules
git submodule update --init --recursive 2>>$logfile 1>>$logfile

if [[ $OSTYPE == 'darwin'* ]]; then
  echo "MacOS"
    FILE_NAME="install_brew_requirements.sh"
else
  source /etc/os-release
  echo $NAME
  FILE_NAME="install_apt_requirements.sh"

  if [ "${NAME}" == "Fedora Linux" ]; then
   FILE_NAME="install_dnf_requirements.sh"
  fi
fi

printf "Check requirements. Running ${FILE_NAME} ...\n"
./scripts/$FILE_NAME
MR_THIRDPARTY_DIR="thirdparty/"

#build Third party
if ! [ -d "./lib/" ]; then
 mkdir -p lib
fi
cd lib
cmake ../${MR_THIRDPARTY_DIR}
make -j `nproc` #VERBOSE=1
cd ..

printf "\rThirdparty build script successfully finished. Required libs located in ./lib folder. You could run ./scripts/build_source.sh\n\n"
