#!/bin/bash
# builds MRBenchCollide once per variant (source/MRBenchCollide/variants/<v> holds that variant's MRMesh files),
# copying the output folder to bin-<v> and the assembly of MRMeshCollidePrecise.cpp to asm/<v>.s;
# extra arguments go to cmake configure
set -euo pipefail
BUILD=build/bench
VARIANTS=${VARIANTS:-master branchless}
NPROC=${NUMBER_OF_PROCESSORS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu)}
PY=; for c in python3 python; do if "$c" -c "" >/dev/null 2>&1; then PY=$c; break; fi; done
OPTS=()
if [ -n "${EXTRA_CMAKE_RELEASE_FLAGS:-}" ]; then OPTS+=(-D "CMAKE_CXX_FLAGS_RELEASE=${EXTRA_CMAKE_RELEASE_FLAGS}"); fi
if command -v ninja >/dev/null 2>&1 ; then OPTS+=(-G Ninja); fi
if [ -n "${MESHLIB_THIRDPARTY_ROOT_DIR:-}" ] && [ -d "${MESHLIB_THIRDPARTY_ROOT_DIR}" ]; then
  OPTS+=(-D "CMAKE_PREFIX_PATH=${MESHLIB_THIRDPARTY_ROOT_DIR}" -D "MESHLIB_THIRDPARTY_ROOT_DIR=${MESHLIB_THIRDPARTY_ROOT_DIR}")
fi
cmake -S . -B "$BUILD" -D CMAKE_BUILD_TYPE=Release -D CMAKE_EXPORT_COMPILE_COMMANDS=ON "${OPTS[@]}" \
  -D MESHLIB_BUILD_MRVIEWER=OFF -D MESHLIB_BUILD_MESHVIEWER=OFF -D MESHLIB_BUILD_PYTHON_MODULES=OFF \
  -D MESHLIB_BUILD_MESHCONV=OFF -D MESHLIB_BUILD_SYMBOLMESH=OFF -D MESHLIB_BUILD_VOXELS=OFF \
  -D MESHLIB_BUILD_EXTRA_IO_FORMATS=OFF -D MESHLIB_BUILD_MCP=OFF -D MESHLIB_BUILD_MRCUDA=OFF \
  -D MESHLIB_PYTHON_SUPPORT=OFF -D BUILD_TESTING=OFF -D CMAKE_INSTALL_INCLUDEDIR=include -D CMAKE_INSTALL_LIBDIR=lib \
  -D MESHLIB_BUILD_GENERATED_C_BINDINGS=OFF "$@"
mkdir -p asm
for v in $VARIANTS; do
  cp -R "source/MRBenchCollide/variants/$v/." .
  cmake --build "$BUILD" --target MRBenchCollide -j "$NPROC"
  rm -rf "bin-$v"
  cp -R "$BUILD/bin" "bin-$v"
  { "$PY" source/MRBenchCollide/asm_dump.py "$BUILD/compile_commands.json" source/MRMesh/MRMeshCollidePrecise.cpp "asm/$v.s" || echo "asm dump failed for $v"; } | tee -a asm/counts.txt
done
ls -la asm
