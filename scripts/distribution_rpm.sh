#!/bin/bash

# This script creates `*.rpm` packages with built thirdparty and project libs
# usage: first argument - `v1.2.3.4` - version with "v" prefix
# ./distribution_rpm.sh v1.2.3.4

# exit if any command failed
set -eo pipefail

if [ ! -f "./lib/libcpr.so" ]; then
  echo "Thirdparty build was not found. Building..."
  ./scripts/build_thirdparty.sh
fi

if [ ! -f "./build/Release/bin/libMRMesh.so" ]; then
  echo "Project release build was not found. Building..."
  export MESHLIB_BUILD_RELEASE="ON"
  export MESHLIB_BUILD_DEBUG="OFF"
  ./scripts/build_source.sh
fi

#modify mr.version
version=0.0.0.0
if [ ${1} ]; then
  version=${1:1} #v1.2.3.4 -> 1.2.3.4
fi
echo $version > build/Release/bin/mr.version

BASE_DIR=$(realpath $(dirname "$0")/..)
REQUIREMENTS_FILE="${BASE_DIR}/requirements/fedora.txt"
# convert multi-line file to comma-separated string
REQUIRES_LINE=$(cat ${REQUIREMENTS_FILE} | tr '\n' ',' | sed -e "s/,\s*$//" -e "s/,/, /g")

# modify rpm spec file
sed -i \
  -e "s/Version:/Version:        ${version}/" \
  -e "s/Requires:/Requires:       ${REQUIRES_LINE}/" \
  ./scripts/MeshLib-dev.spec

#create distr dirs
if [ -d "./rpmbuild/" ]; then
 rm -rf rpmbuild
fi

mkdir -p rpmbuild/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
rpmbuild -bb .${MRE_PREFIX}/scripts/MeshLib-dev.spec  --define "_topdir `pwd`/rpmbuild" --target x86_64
mv rpmbuild/RPMS/meshlib-dev.rpm .

rm -rf rpmbuild
