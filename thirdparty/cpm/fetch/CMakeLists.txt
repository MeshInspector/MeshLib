cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Materializes dependency sources without building them, and writes their locations to
# MESHLIB_FETCH_OUTPUT as `SRC_<name>=<path>` lines -- a form both `sh` and `make` can read.
# scripts/mrbind/generate.mk uses this to parse MeshLib headers against raw dependency sources
# on platforms that never run scripts/build_thirdparty.sh.
#
# project(... NONE) skips compiler detection, so this configures with no toolchain file even
# when the caller is targeting Emscripten.
project(MeshLib_thirdparty_fetch NONE)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/CPM.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../package-lock.cmake)

foreach(pkg IN LISTS MESHLIB_FETCH_PACKAGES)
  CPMAddPackage(NAME ${pkg} ${MESHLIB_PACKAGE_${pkg}} DOWNLOAD_ONLY YES)
  # Hyphenated package names are not legal shell or make variable names.
  string(MAKE_C_IDENTIFIER "${pkg}" var)
  string(APPEND sources "SRC_${var}=${${pkg}_SOURCE_DIR}\n")
endforeach()

file(WRITE "${MESHLIB_FETCH_OUTPUT}" "${sources}")
