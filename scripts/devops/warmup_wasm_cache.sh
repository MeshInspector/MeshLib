#!/bin/bash

# Warms emcc's cache for the Emscripten configuration currently selected by
# MR_EMSCRIPTEN_SINGLE / MR_EMSCRIPTEN_WASM64, so that no build has to generate
# system libraries or download the -sUSE_* ports inside cmake's first
# try_compile. Run it right after scripts/build_thirdparty.sh, which selects the
# same configuration.
#
# The flags are not repeated here: the stub below includes DefaultOptions and
# ConfigureEmscripten, the modules MeshLib itself includes, so it follows them.
# Release covers what the build links; Debug covers what try_compile links,
# which is unoptimised and so selects the assertions-enabled variants.

set -eo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MESHLIB_ROOT="$(realpath "${SCRIPT_DIR}/../..")"

. "${SCRIPT_DIR}/../ask_emscripten_mode.src"

if [ "${MR_EMSCRIPTEN}" != "ON" ] ; then
  echo "Not an Emscripten build, nothing to warm up"
  exit 0
fi

WARMUP_DIR="$(mktemp -d)"
trap 'rm -rf "${WARMUP_DIR}"' EXIT

cat > "${WARMUP_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# must precede project(): it carries the MEMORY64 flags platform detection needs
include(DefaultOptions)

project(MeshLibEmscriptenCacheWarmup CXX)

# ConfigureEmscripten reads this to decide on the MRViewer link flags, which
# select the libGL variant; MeshLib's own CMakeLists defaults it to ON.
option(MESHLIB_BUILD_MRVIEWER "Build MRViewer library" ON)

include(ConfigureEmscripten)

add_executable(warmup main.cpp)
EOF

cat > "${WARMUP_DIR}/main.cpp" <<'EOF'
int main() { return 0; }
EOF

WARMUP_CMAKE_OPTIONS="\
  -D CMAKE_MODULE_PATH=${MESHLIB_ROOT}/cmake/Modules \
  -D MR_EMSCRIPTEN=1 \
  -D MR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLETHREAD} \
  -D MR_EMSCRIPTEN_WASM64=${MR_EMSCRIPTEN_WASM64} \
"
if [ -n "${MR_EMSCRIPTEN_MIMALLOC}" ] ; then
  WARMUP_CMAKE_OPTIONS="${WARMUP_CMAKE_OPTIONS} -D MR_EMSCRIPTEN_MIMALLOC=${MR_EMSCRIPTEN_MIMALLOC}"
fi

for WARMUP_CONFIG in Release Debug ; do
  emcmake cmake -G Ninja -S "${WARMUP_DIR}" -B "${WARMUP_DIR}/build" \
    -D CMAKE_BUILD_TYPE=${WARMUP_CONFIG} ${WARMUP_CMAKE_OPTIONS}
  cmake --build "${WARMUP_DIR}/build"
  rm -rf "${WARMUP_DIR}/build"
done
