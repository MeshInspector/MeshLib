#!/bin/bash

# Warms emcc's cache so that no build generates system libraries or downloads
# the -sUSE_* ports inside cmake's first try_compile. Run right after
# scripts/build_thirdparty.sh, whose configuration it reuses.

set -eo pipefail

WARMUP_DIR="$(mktemp -d)"
trap 'rm -rf "${WARMUP_DIR}"' EXIT

# The flags are not repeated here: this stub includes the same modules MeshLib's
# own CMakeLists does, so it follows them.
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

MESHLIB_ROOT="$(realpath "$(dirname "$(realpath "$0")")/../..")"

WARMUP_CMAKE_OPTIONS="\
  -D CMAKE_MODULE_PATH=${MESHLIB_ROOT}/cmake/Modules \
  -D MR_EMSCRIPTEN=1 \
  -D MR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLE:-OFF} \
  -D MR_EMSCRIPTEN_WASM64=${MR_EMSCRIPTEN_WASM64:-OFF} \
"
if [ -n "${MR_EMSCRIPTEN_MIMALLOC}" ] ; then
  WARMUP_CMAKE_OPTIONS="${WARMUP_CMAKE_OPTIONS} -D MR_EMSCRIPTEN_MIMALLOC=${MR_EMSCRIPTEN_MIMALLOC}"
fi

# Release is what the build links, Debug what try_compile links: it links
# without optimisation and so selects the assertions-enabled variants.
for WARMUP_CONFIG in Release Debug ; do
  emcmake cmake -G Ninja -S "${WARMUP_DIR}" -B "${WARMUP_DIR}/build" \
    -D CMAKE_BUILD_TYPE=${WARMUP_CONFIG} ${WARMUP_CMAKE_OPTIONS}
  cmake --build "${WARMUP_DIR}/build"
  rm -rf "${WARMUP_DIR}/build"
done
