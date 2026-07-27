#!/bin/bash

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. \
    thirdparty/imgui \
    thirdparty/eigen \
    thirdparty/parallel-hashmap \
    thirdparty/mrbind-pybind11 \
    thirdparty/mrbind \

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl
