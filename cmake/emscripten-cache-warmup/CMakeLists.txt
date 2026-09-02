# Links a trivial program with the very flags a MeshLib Emscripten build uses,
# so a docker image can populate emcc's cache at image build time. emcc caches
# system libraries and ports per flag combination, and cmake's first
# try_compile already links with the full set, so an unwarmed cache makes every
# CI job build libc & co and download the -sUSE_* ports before configuring.
#
# The flags are deliberately NOT repeated here: DefaultOptions and
# ConfigureEmscripten are the same modules the real build includes, so this
# project follows them automatically. Configure it exactly as
# scripts/build_source.sh configures MeshLib, e.g.
#
#   emcmake cmake -S cmake/emscripten-cache-warmup -B <build dir> \
#     -D CMAKE_BUILD_TYPE=Release -D MR_EMSCRIPTEN=1 \
#     -D MR_EMSCRIPTEN_SINGLETHREAD=0 -D MR_EMSCRIPTEN_WASM64=0
#
# Build it once per configuration the CI uses, in both Release and Debug:
# try_compile links without optimisation and so picks the assertions-enabled
# `-debug` variants, while the build itself links the others.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../Modules")

# must precede project(): it carries the MEMORY64 flags platform detection needs
include(DefaultOptions)

project(MeshLibEmscriptenCacheWarmup CXX)

IF(NOT EMSCRIPTEN)
  message(FATAL_ERROR "This project is only meaningful for an Emscripten build")
ENDIF()

# ConfigureEmscripten reads this to decide on the MRViewer link flags, which
# select the libGL variant; MeshLib's own CMakeLists defaults it to ON.
option(MESHLIB_BUILD_MRVIEWER "Build MRViewer library" ON)

include(ConfigureEmscripten)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/warmup.cpp" "int main() { return 0; }\n")
add_executable(warmup "${CMAKE_CURRENT_BINARY_DIR}/warmup.cpp")
