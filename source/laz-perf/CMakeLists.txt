cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# This file exists so we can disable some warnings in Lazperf.
# If we were to `add_subdirectory` it directly from `MeshLib/CMakeLists.txt`, we would have to modify the global CXX flags, which is uncool.

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4245 /wd4456 /wd4458 /wd5051")
set(WITH_TESTS OFF CACHE BOOL "Choose if LAZPERF unit tests should be built")
add_subdirectory(${MESHLIB_THIRDPARTY_DIR}/laz-perf ./laz-perf)
