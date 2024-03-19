#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(dirname $(realpath "$0"))
PATCH_DIR="${SCRIPT_DIR}/patches/"
SOURCE_DIR="${SCRIPT_DIR}/../"
BUILD_DIR="${SCRIPT_DIR}/../build/"
INSTALL_DIR="$1"

PATCHES=(
  occt-dependencies.patch
)
for PATCH in $PATCHES ; do
  # check if the patch is already applied
  OUTPUT=$(patch \
    --dry-run \
    --quiet \
    --batch \
    --reverse \
    --strip=1 \
    --directory="$SOURCE_DIR/occt/" \
    --input="$PATCH_DIR/$PATCH"
  )
  if [ -n "$OUTPUT" ] ; then
    patch \
      --quiet \
      --batch \
      --forward \
      --reject-file=- \
      --strip=1 \
      --directory="$SOURCE_DIR/occt/" \
      --input="$PATCH_DIR/$PATCH"
  fi
done

cmake -S "$SOURCE_DIR/occt/" -B "$BUILD_DIR/occt/" \
  -DUSE_FREETYPE=OFF \
  -DUSE_FREEIMAGE=OFF \
  -DUSE_OPENVR=OFF \
  -DUSE_OPENGL=OFF \
  -DUSE_GLES2=OFF \
  -DUSE_RAPIDJSON=OFF \
  -DUSE_DRACO=OFF \
  -DUSE_TK=OFF \
  -DUSE_TBB=ON \
  -DUSE_VTK=OFF \
  -DUSE_XLIB=OFF \
  -DBUILD_MODULE_Draw=OFF \
  -DBUILD_MODULE_Visualization=OFF \
  -DBUILD_MODULE_ApplicationFramework=OFF \
  -DBUILD_MODULE_DETools=OFF \
  -DBUILD_MODULE_ModelingAlgorithms=OFF \
  -DBUILD_MODULE_ModelingData=OFF \
  -DBUILD_MODULE_DataExchange=OFF \
  -DBUILD_RELEASE_DISABLE_EXCEPTIONS=ON \
  -DBUILD_Inspector=OFF \
  -DBUILD_DOC_Overview=OFF \
  -DBUILD_ADDITIONAL_TOOLKITS='TKDESTEP;TKBinXCAF' \
  -DBUILD_CPP_STANDARD=C++20 \
  -DBUILD_OPT_PROFILE=Production \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

cmake --build "$BUILD_DIR/occt/" \
  --config Release \
  --target install \
  -j$(nproc)
