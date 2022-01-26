cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (MeshRUs_thirdparty CXX)

# all binaries will be located in ./build/Release/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare -Wno-unknown-pragmas \
	-Wno-unused-function -Wno-unused-variable -Wunused-parameter -Wno-comment -Wno-parentheses \
	-Wredundant-move -Wreturn-type \
	-Wno-int-in-bool-context -Wno-strict-aliasing")
# -Werror")

#add_compile_definitions(__emscripten__)
get_filename_component(TBB_ROOT "./wasmtbb" ABSOLUTE)
#include(${TBB_ROOT}/cmake/TBBBuild.cmake)
#tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR)
#find_package(TBB REQUIRED)
#add_subdirectory(./wasmtbb ./wasmtbb)
add_custom_target(
   tbb
   COMMAND make
   WORKING_DIRECTORY ${TBB_ROOT}
)

add_subdirectory(./googletest/googletest ./googletest)

#IF(APPLE)
 add_subdirectory(./expected ./expected)
#ELSE()
 # set platform dependent flags
# file(STRINGS /etc/os-release distro REGEX "^NAME=")
# string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" distro "${distro}")
# IF( NOT ${distro} STREQUAL "Fedora Linux")
#  set(OPENVDB_ENABLE_UNINSTALL OFF)
#  set(OPENVDB_ENABLE_INSTALL OFF)
#  add_subdirectory(./openvdb ./openvdb)
# ENDIF()

 #set(CPR_ENABLE_UNINSTALL "OFF")
 #set(CPR_ENABLE_INSTALL "OFF")
 #set(CPR_USE_SYSTEM_GTEST "ON")
 #set(CPR_BUILD_TESTS "OFF")
 #set(CPR_BUILD_TESTS_SSL "OFF")
 #add_subdirectory(./cpr ./cpr)
#ENDIF()

add_subdirectory(./OpenCTM ./OpenCTM)
add_subdirectory(./parallel-hashmap ./parallel-hashmap)
