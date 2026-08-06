#!/bin/bash

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

SUBMODULES=(
    thirdparty/imgui
    thirdparty/eigen
    thirdparty/mrbind-pybind11
    thirdparty/mrbind
)

if [[ $1 != --skip-prebuilt-thirdparty ]]; then
    SUBMODULES+=(
        thirdparty/clip
        thirdparty/cpp-httplib
        thirdparty/cpr
        thirdparty/fastmcpp
        thirdparty/glad
        thirdparty/laz-perf
        thirdparty/libE57Format
        thirdparty/nlohmann-json
        thirdparty/OpenCTM-git
        thirdparty/openvdb/v10/openvdb
        thirdparty/parallel-hashmap
        thirdparty/tinygltf
        thirdparty/zlib-ng
    )
fi

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. "${SUBMODULES[@]}"
"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl
