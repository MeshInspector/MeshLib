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

# Bundle Homebrew dylib deps into the framework so the .pkg is robust against
# bottle SONAME drift (e.g. jsoncpp dropping libjsoncpp.27.dylib). System libs
# and libpython are left as external references on purpose. Uses CMake's
# BundleUtilities::fixup_bundle() via scripts/fixup_bundle_macos.cmake.
APP="$(pwd)/${FRAMEWORK_DIR#./}/bin/MeshViewer"

# Enumerate every other Mach-O already shipped in the framework. They get
# passed to fixup_bundle via LIBS so it treats them as in-bundle items
# (and walks their prereqs) instead of trying to re-copy them. Mach-O
# detection via `file -b` matches both executables (Mach-O ... executable)
# and dynamic libs / Python .so bundles (Mach-O ... dynamically linked
# shared library / bundle).
LIBS_LIST=()
while IFS= read -r f ; do
  if file -b "$f" | grep -qi "Mach-O" ; then
    [ "$f" = "$APP" ] || LIBS_LIST+=("$f")
  fi
done < <(find "${FRAMEWORK_DIR}/bin" "${FRAMEWORK_DIR}/lib" -type f)
LIBS_SEMI="$(IFS=';' ; echo "${LIBS_LIST[*]}")"

# Search dirs for resolving @rpath/... prereqs. Built from the active
# Homebrew prefix (`brew --prefix`) so the arm64 self-hosted runner whose
# prefix is /Users/runner/.homebrew works alongside the github-hosted
# /usr/local install.
BREW_PREFIX="$(brew --prefix)"
DIRS_LIST=("${BREW_PREFIX}/lib")
for d in "${BREW_PREFIX}/opt"/*/lib ; do
  [ -d "$d" ] && DIRS_LIST+=("$d")
done
for d in "${BREW_PREFIX}/Cellar"/*/*/lib ; do
  [ -d "$d" ] && DIRS_LIST+=("$d")
done
DIRS_SEMI="$(IFS=';' ; echo "${DIRS_LIST[*]}")"

cmake \
  -DAPP="${APP}" \
  -DLIBS="${LIBS_SEMI}" \
  -DDIRS="${DIRS_SEMI}" \
  -P scripts/fixup_bundle_macos.cmake

# install_name_tool invalidates the linker-embedded ad-hoc signatures
# fixup_bundle relies on; arm64 macOS SIGKILLs any such binary on launch.
# Re-sign every Mach-O ad-hoc, preserving entitlements / runtime flags
# the linker put there. Proper release signing (if any) happens later.
while IFS= read -r f ; do
  if file -b "$f" | grep -qi "Mach-O" ; then
    codesign --force --sign - \
      --preserve-metadata=entitlements,requirements,flags,runtime "$f"
  fi
done < <(find "${FRAMEWORK_DIR}/bin" "${FRAMEWORK_DIR}/lib" -type f)

# FIXME: this breaks CMake config
#pushd "${FRAMEWORK_BASE_DIR}"
#  ln -s "Versions/${VERSION}/Resources" Resources
#popd

pushd "${FRAMEWORK_BASE_DIR}/Versions"
  ln -s "${VERSION}" Current
popd

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
