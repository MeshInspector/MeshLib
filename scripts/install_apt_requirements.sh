#!/bin/bash

# NODE: using this script is deprecated! Better install meshlib(-dev).deb package
# This script installs requirements by `apt` if not already installed
# `distribution.sh` uses this script as preinstall

ALL_REQUIRED_PACKAGES_INSTALLED=true
MISSED_PACKAGES=""
function checkPackage {
  PKG_OK=$(dpkg-query -W --showformat='${Status}\n' ${1} | grep "install ok installed")
  if [ "" = "$PKG_OK" ]; then
    ALL_REQUIRED_PACKAGES_INSTALLED=false
    MISSED_PACKAGES="${MISSED_PACKAGES} ${1}"
  fi
}

BASEDIR=$(dirname $(realpath "$0"))
REQUIREMENTS_FILE="$BASEDIR"/../requirements/ubuntu.txt

for req in `cat $REQUIREMENTS_FILE`; do
  checkPackage "${req}"
done

if $ALL_REQUIRED_PACKAGES_INSTALLED; then
  echo "All required packages are already installed!"
  exit 0
fi

echo "Some required packages are not installed!"
echo "${MISSED_PACKAGES}"

if [ "$EUID" -ne 0 ]; then
  echo "Root access required!"
  RUN_AS_ROOT="NO"
fi

if [ "$MR_STATE" != "DOCKER_BUILD" ]; then
  sudo apt-get update && \
  sudo apt-get install ${MISSED_PACKAGES}
else
  sudo apt-get -y update && \
  sudo apt-get -y install ${MISSED_PACKAGES}
fi

# check and upgrade python3 pip
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip

# install requirements for python libs
python3 -m pip install -r requirements/python.txt

# invalidate sudo credentials
if [ "${RUN_AS_ROOT}" = "NO" ]; then
  sudo -k
fi

exit 0
