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

BASEDIR=$(dirname "$0")
requirements_file="$BASEDIR"/../requirements/ubuntu.txt
. /etc/lsb-release
if [ "$DISTRIB_ID" == "Ubuntu" ] && [ "$DISTRIB_RELEASE" == "22.04" ]; then
  requirements_file="$BASEDIR"/../requirements/ubuntu22.txt
fi

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
 sudo apt update && sudo apt install ${MISSED_PACKAGES}
else
 sudo apt-get -y update && sudo apt-get -y  install ${MISSED_PACKAGES}
fi

. /etc/lsb-release
if [ "$DISTRIB_ID" == "Ubuntu" ] && [ "$DISTRIB_RELEASE" == "22.04" ]; then
  python3.10 -m ensurepip --upgrade
  python3.10 -m pip install --upgrade pip

  # install requirements for python libs
  python3.10 -m pip install -r requirements/python.txt
else
  # check and upgrade python3 pip
  python3.8 -m ensurepip --upgrade
  python3.8 -m pip install --upgrade pip

  # install requirements for python libs
  python3.8 -m pip install -r requirements/python.txt
fi
# fix boost signal2 C++20 error in default version 1.71.0 from `apt`
# NOTE: 1.75+ version already has this fix
# https://github.com/boostorg/signals2/commit/15fcf213563718d2378b6b83a1614680a4fa8cec
FILENAME=/usr/include/boost/signals2/detail/auto_buffer.hpp
cat $FILENAME | tr '\n' '\r' | \
sed -e 's/\r        typedef typename Allocator::pointer              allocator_pointer;\r/\
#ifdef BOOST_NO_CXX11_ALLOCATOR\
        typedef typename Allocator::pointer allocator_pointer;\
#else\
        typedef typename std::allocator_traits<Allocator>::pointer allocator_pointer;\
#endif\
/g' | tr '\r' '\n' | sudo tee $FILENAME

if [ "${RUN_AS_ROOT}" = "NO" ]; then
 sudo -k
fi

exit 0
