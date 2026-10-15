#!/bin/bash

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

SUBMODULES=(
    thirdparty/imgui
    thirdparty/mrbind
    thirdparty/mrbind-pybind11
)

# with --vcpkg the remaining thirdparty comes from vcpkg ports, otherwise build_thirdparty.sh builds it
if [[ $1 != --vcpkg ]]; then
    SUBMODULES+=(
        thirdparty/clip
        thirdparty/draco
        thirdparty/fastmcpp
        thirdparty/glad
        thirdparty/laz-perf
        thirdparty/libE57Format
        thirdparty/OpenCTM-git
        thirdparty/tinygltf
    )
fi

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. "${SUBMODULES[@]}"
"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl
