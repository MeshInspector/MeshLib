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

add_subdirectory(./googletest/googletest ./googletest)

add_subdirectory(./libzip ./libzip)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s NO_DISABLE_EXCEPTION_CATCHING=1")
add_subdirectory(./jsoncpp ./jsoncpp)