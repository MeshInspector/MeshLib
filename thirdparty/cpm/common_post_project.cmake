# Shared stage setup that must run after project(); see common_pre_project.cmake for why the
# two halves are separate files.

# all binaries will be located in ./build/Release/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Inhibit all warning messages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")

IF(MR_EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
  IF(NOT MR_EMSCRIPTEN_SINGLETHREAD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
  ENDIF()
ENDIF()

# CPM.cmake reissues cmake_minimum_required(3.14), so include it after project() to keep that
# out of the policy scope the stage files are configured under.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CPM.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/package-lock.cmake)

macro(meshlib_add_package name)
  CPMAddPackage(NAME ${name} ${MESHLIB_PACKAGE_${name}} ${ARGN})
endmacro()
