#!/bin/bash

git -C "$(dirname "$BASH_SOURCE")"/.. submodule update --init --depth 1 \
    thirdparty/imgui \
    thirdparty/parallel-hashmap \
    thirdparty/mrbind-pybind11 \
    thirdparty/mrbind \

git -C "$(dirname "$BASH_SOURCE")"/../thirdparty/mrbind submodule update --init --depth 1 deps/cppdecl
