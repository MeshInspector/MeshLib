#!/bin/bash

# Submodules that build_thirdparty.sh builds for Emscripten (the Emscripten Docker images).

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. \
    thirdparty/c-blosc \
    thirdparty/draco \
    thirdparty/eigen \
    thirdparty/expected \
    thirdparty/fmt \
    thirdparty/GDCM \
    thirdparty/glad \
    thirdparty/googletest \
    thirdparty/jsoncpp \
    thirdparty/laz-perf \
    thirdparty/libE57Format \
    thirdparty/libjpeg-turbo \
    thirdparty/libzip \
    thirdparty/mbedtls \
    thirdparty/onetbb \
    thirdparty/OpenCTM-git \
    thirdparty/openvdb/v10/openvdb \
    thirdparty/parallel-hashmap \
    thirdparty/spdlog \
    thirdparty/tinygltf \
    thirdparty/tinyxml2 \
    thirdparty/zlib-ng
