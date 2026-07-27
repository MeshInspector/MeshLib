#!/bin/bash

git -C "$(dirname "$BASH_SOURCE")"/.. submodule update --init --depth 1 \
    thirdparty/eigen \
    thirdparty/expected \
    thirdparty/imgui \
    thirdparty/mrbind \
    thirdparty/mrbind-pybind11 \
    thirdparty/parallel-hashmap \

git -C "$(dirname "$BASH_SOURCE")"/../thirdparty/mrbind submodule update --init --depth 1 deps/cppdecl
