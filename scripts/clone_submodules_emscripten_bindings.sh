#!/bin/bash

# This is to be used on platforms that generate C/C# bindings through Emscripten: on Windows, and optionally on Linux. Not needed on Mac.

set -e

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. \
    thirdparty/eigen \
    thirdparty/expected \
    thirdparty/imgui \
    thirdparty/jsoncpp \
    thirdparty/mrbind \
    thirdparty/mrbind-pybind11 \
    thirdparty/onetbb \
    thirdparty/openvdb/v10/openvdb \
    thirdparty/parallel-hashmap \
    thirdparty/spdlog \

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl
