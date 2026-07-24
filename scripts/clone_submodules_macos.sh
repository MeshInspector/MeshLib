#!/bin/bash

git -C "$(dirname "$BASH_SOURCE")"/.. submodule update --init --depth 1 \
    thirdparty/imgui \
    thirdparty/eigen \
    thirdparty/parallel-hashmap \
    thirdparty/expected \
    thirdparty/OpenCTM-git \
    thirdparty/libE57Format \
    thirdparty/glad \
    thirdparty/tinygltf \
    thirdparty/laz-perf \
    thirdparty/clip \
    thirdparty/fastmcpp \
    thirdparty/nlohmann-json \
    thirdparty/cpp-httplib \
    thirdparty/mrbind \
    thirdparty/mrbind-pybind11 \

git -C "$(dirname "$BASH_SOURCE")"/../thirdparty/mrbind submodule update --init --depth 1 deps/cppdecl
