#!/bin/bash

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

SUBMODULES=(
    thirdparty/clip
    thirdparty/cpp-httplib
    thirdparty/eigen
    thirdparty/expected
    thirdparty/fastmcpp
    thirdparty/glad
    thirdparty/imgui
    thirdparty/laz-perf
    thirdparty/libE57Format
    thirdparty/mrbind
    thirdparty/mrbind-pybind11
    thirdparty/nlohmann-json
    thirdparty/OpenCTM-git
    thirdparty/tinygltf
)

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. "${SUBMODULES[@]}"
"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl
