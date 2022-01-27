cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


IF(APPLE)
 message("building for Apple")
 set(PYTHON_LIB python3.9)
 execute_process(
         COMMAND brew --prefix
         RESULT_VARIABLE CMD_ERROR
         OUTPUT_VARIABLE HOMEBREW_PREFIX
         OUTPUT_STRIP_TRAILING_WHITESPACE)
 IF(CMD_ERROR EQUAL 0 AND EXISTS "${HOMEBREW_PREFIX}")
  message("Homebrew found. Prefix:${HOMEBREW_PREFIX} ")
 ELSE()
  message("Homebrew not found!")
  message( FATAL_ERROR "${CMD_ERROR} ${HOMEBREW_PREFIX}" )
 ENDIF()
 # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
 LINK_DIRECTORIES(${HOMEBREW_PREFIX}/lib)
 LINK_DIRECTORIES(${HOMEBREW_PREFIX}/opt/python@3.9/Frameworks/Python.framework/Versions/3.9/lib/)
 set(CPPFLAGS "-I${HOMEBREW_PREFIX}/opt/llvm/include -I${HOMEBREW_PREFIX}/include")
 set(LDFLAGS "-L${HOMEBREW_PREFIX}/opt/llvm/lib -Wl,-rpath,${HOMEBREW_PREFIX}/opt/llvm/lib")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++20-stdlib=libc++ -undefined dynamic_lookup -framework Cocoa -framework IOKit") # https://github.com/pybind/pybind11/issues/382
ELSE()
 set(PYTHON_LIB python3.8)
 # print compile time of each translation unit,
 # such print format is selected to simplify load in Excel with fields (user time, system time, command) subdivision by commas
 set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "/usr/bin/time -f \",%U,%S,%C\"")
 set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "/usr/bin/time -f \",%U,%S,%C\"")
ENDIF()

project (MeshLib CXX)
add_compile_definitions(MR_PROJECT_NAME=\"MeshLib\")

# all binaries will be located in ./build/Release/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DDEBUG")
# turn on warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wno-sign-compare -Werror -fvisibility=hidden -pedantic-errors")

set(MESHLIB_THIRDPARTY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/thirdparty")
set(THIRDPARTY_LIB_DIR "${PROJECT_SOURCE_DIR}/lib")

IF(NOT EXISTS "${THIRDPARTY_LIB_DIR}")
 message(FATAL_ERROR "thirdparty build directory not found! You can build thirdparty with ./scripts/build_thirdparty.sh")
ENDIF()
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR})
link_directories(${THIRDPARTY_LIB_DIR})

find_package(PkgConfig REQUIRED)

IF(APPLE)
 include_directories(${HOMEBREW_PREFIX}/include)

 find_package(pybind11  REQUIRED)
 find_package(Python 3.9 EXACT REQUIRED)
 find_package(PythonInterp 3.9 EXACT REQUIRED)
 find_package(PythonLibs 3.9 EXACT REQUIRED)
 include_directories(${PYTHON_INCLUDE_DIRS})
ELSE()
 include_directories(/usr/include)
 set(PYBIND11_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/pybind11/include)
 include_directories(${PYBIND11_DIR})
 find_package(PythonLibs 3.8 EXACT REQUIRED)
ENDIF()
include_directories(${PYTHON_INCLUDE_PATH})

set(PROJECT_SOURCE_DIR ./source)
include_directories(${PROJECT_SOURCE_DIR})

enable_testing()

IF(NOT APPLE)
 # set platform dependent flags
 file(STRINGS /etc/os-release distro REGEX "^NAME=")
 string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" distro "${distro}")
 message("using ${distro}")
 IF(NOT ${distro} STREQUAL "Fedora Linux")
  set(OPENVDB_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/openvdb/openvdb)
  include_directories(${OPENVDB_DIR})
 ELSE()
  add_compile_definitions(SPDLOG_FMT_EXTERNAL)
 ENDIF()
ENDIF()
add_compile_definitions(SPDLOG_COMPILED_LIB SPDLOG_SHARED_LIB)

add_subdirectory(${PROJECT_SOURCE_DIR}/MRMesh ./MRMesh)
add_subdirectory(${PROJECT_SOURCE_DIR}/MREAlgorithms ./MREAlgorithms)
add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshpy ./mrmeshpy)
add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshnumpy ./mrmeshnumpy)
add_subdirectory(${PROJECT_SOURCE_DIR}/mrealgorithmspy ./mrealgorithmspy)
add_subdirectory(${PROJECT_SOURCE_DIR}/MRTest ./MRTest)
