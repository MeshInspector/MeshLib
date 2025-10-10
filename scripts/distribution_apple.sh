#!/bin/bash

set -euxo pipefail

VERSION=${1:-v0.0.0.0}
VERSION=${VERSION:1}  # v1.2.3.4 -> 1.2.3.4

if [ -d ./macos_distr ] ; then
  rm -rf ./macos_distr
fi
mkdir ./macos_distr

FRAMEWORK_BASE_DIR="./macos_distr/Library/Frameworks/MeshLib.framework"
FRAMEWORK_DIR="${FRAMEWORK_BASE_DIR}/Versions/${VERSION}"

cmake --install build/Release --prefix="${FRAMEWORK_DIR}"
echo "version: ${VERSION}"
echo "prefix: ${FRAMEWORK_DIR}"

cp -rL ./lib "${FRAMEWORK_DIR}/lib/"
cp -rL ./include "${FRAMEWORK_DIR}/include/"

cp ./LICENSE ./macos/Resources
mkdir "${FRAMEWORK_DIR}/requirements/"
cp ./requirements/macos.txt "${FRAMEWORK_DIR}/requirements/"

ln -s "/Library/Frameworks/MeshLib.framework/Versions/${VERSION}" "${FRAMEWORK_BASE_DIR}/Versions/Current"
ln -s "/Library/Frameworks/MeshLib.framework/Resources"           "${FRAMEWORK_BASE_DIR}/Versions/${VERSION}/Resources"

# be careful with pkg names! The pkg can fail to build
pkgbuild \
  --root macos_distr/Library \
  --identifier com.MeshInspector.MeshLib \
  --install-location /Library \
  MeshLib.pkg

productbuild \
  --distribution ./macos/Distribution.xml \
  --package-path ./MeshLib.pkg \
  --resources ./macos/Resources \
  MeshLib_.pkg

rm -r ./macos_distr
