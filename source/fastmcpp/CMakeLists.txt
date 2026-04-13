# This file exists so we can tweak some settings for Fastmcpp.
# If we were to `add_subdirectory` it directly from `CMakeLists.txt`, we would have to modify the global CXX flags, which is uncool.

# Sync those flags with `scripts/build_thirdparty.sh`.

IF(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF()

IF(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4099 /wd4100 /wd4242 /wd4244 /wd4355 /wd4456 /wd4458 /wd4464 /wd4505 /wd4702 /wd5204 /wd5220 /wd5233 /wd5245")
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-error")
ENDIF()

set(FASTMCPP_BUILD_TESTS OFF CACHE BOOL "Build tests")
set(FASTMCPP_BUILD_EXAMPLES OFF CACHE BOOL "Build examples")
set(FASTMCPP_FETCH_CURL OFF CACHE BOOL "Fetch and build libcurl statically for POST streaming")

add_subdirectory(${MESHINSPECTOR_THIRDPARTY_DIR}/fastmcpp fastmcpp)
