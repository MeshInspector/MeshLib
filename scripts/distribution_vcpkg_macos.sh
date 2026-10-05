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
mkdir -p "${FRAMEWORK_DIR}"

cmake --install build/Release --prefix "${FRAMEWORK_DIR}" --strip
echo "version: ${VERSION}"
echo "prefix: ${FRAMEWORK_DIR}"

echo "${VERSION}" > "${FRAMEWORK_DIR}/Resources/mr.version"

cp ./LICENSE ./macos/Resources

pushd "${FRAMEWORK_BASE_DIR}/Versions"
  ln -s "${VERSION}" Current
popd

# drop macOS file metadata
# otherwise pkgbuild adds a hidden ._<name> companion file for every file in the package
xattr -cr ./macos_distr

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
