#!/bin/bash

# run from repo root
# NODE: using this script is deprecated! Better install meshrus(-dev).rpm package
# This script installs requirements by `dnf` if not already installed
# `distribution.sh` uses this script as preinstall

ALL_REQUIRED_PACKAGES_INSTALLED=true
MISSED_PACKAGES=""
function checkPackage {
 PKG_OK=$(rpm -q ${1} | grep "is not installed")
 if [ "" != "$PKG_OK" ]; then
  ALL_REQUIRED_PACKAGES_INSTALLED=false
  MISSED_PACKAGES="${MISSED_PACKAGES} ${1}"
 fi
}

BASEDIR=$(dirname "$0")
requirements_file="$BASEDIR"/../requirements/fedora.txt
for req in `cat $requirements_file`
do
  checkPackage "${req}"
done

if $ALL_REQUIRED_PACKAGES_INSTALLED; then
 printf "\rAll required packages are already installed!                    \n"
 exit 0
fi

printf "\rSome required package(s) are not installed!                     \n"
printf "${MISSED_PACKAGES}\n"

if [ "$EUID" -ne 0 ]; then
 printf "Root access required!\n"
 RUN_AS_ROOT="NO"
fi

if [ $MR_STATE != "DOCKER_BUILD" ]; then
 sudo -s printf "Root access acquired!\n" && \
 sudo dnf update && sudo dnf install ${MISSED_PACKAGES}
else
 sudo dnf -y update && sudo dnf -y  install ${MISSED_PACKAGES}
fi

# check and upgrade python3.9 pip
python3.9 -m ensurepip --upgrade
python3.9 -m pip install --upgrade pip

# install requirements for python libs
python3.9 -m pip install -r requirements/python.txt

exit 0
