build_install () {
  set -eo pipefail

  SOURCE_DIR="$1"
  BUILD_DIR="$2"
  CMAKE_OPTIONS="$3"

  cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" ${CMAKE_OPTIONS}
  # FIXME: build might fail on the first try due to linkage's race condition (?)
  set +e
  cmake --build "${BUILD_DIR}" -j `nproc`
  set -e
  cmake --build "${BUILD_DIR}" -j `nproc`
  cmake --install "${BUILD_DIR}"
}
