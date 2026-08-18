#!/bin/bash

# This is to be used on platforms that generate C/C# bindings through Emscripten: on Windows, and optionally on Linux. Not needed on Mac.

git -C "$(dirname "$BASH_SOURCE")"/.. submodule update --init --depth 1 \
    thirdparty/imgui \
    thirdparty/mrbind \
    thirdparty/mrbind-pybind11 \

git -C "$(dirname "$BASH_SOURCE")"/../thirdparty/mrbind submodule update --init --depth 1 deps/cppdecl
