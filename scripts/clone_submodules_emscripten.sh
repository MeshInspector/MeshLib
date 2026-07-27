#!/bin/bash

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

SUBMODULES=(
    thirdparty/imgui
    thirdparty/mrbind
    thirdparty/mrbind-pybind11
    thirdparty/parallel-hashmap
)

if [[ $1 != --skip-prebuilt-thirdparty ]]; then
    SUBMODULES+=(
        thirdparty/c-blosc
        thirdparty/clip
        thirdparty/expected
        thirdparty/fastmcpp
        thirdparty/fmt
        thirdparty/GDCM
        thirdparty/glad
        thirdparty/googletest
        thirdparty/jsoncpp
        thirdparty/laz-perf
        thirdparty/libzip
        thirdparty/onetbb
        thirdparty/OpenCTM-git
        thirdparty/spdlog
        thirdparty/tinygltf
        thirdparty/tinyxml2
        thirdparty/zlib-ng
    )
fi

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. "${SUBMODULES[@]}"
"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl
