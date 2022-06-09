cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(NOT MR_EMSCRIPTEN)
set(PYTHON_LIB python3.9)
IF(APPLE)
 message("building for Apple")
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
 # uncomment to print compile time of each translation unit,
 # such print format is selected to simplify load in Excel with fields (user time, system time, command) subdivision by commas
 #set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "/usr/bin/time -f \",%U,%S,%C\"")
 #set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "/usr/bin/time -f \",%U,%S,%C\"")
ENDIF() # APPLE
ENDIF() # NOT MR_EMSCRIPTEN

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
set(MESHLIB_THIRDPARTY_LIB_DIR ${THIRDPARTY_LIB_DIR})

IF(NOT EXISTS "${THIRDPARTY_LIB_DIR}")
 message(FATAL_ERROR "thirdparty build directory not found! You can build thirdparty with ./scripts/build_thirdparty.sh")
ENDIF()
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR})
link_directories(${THIRDPARTY_LIB_DIR})

find_package(PkgConfig REQUIRED)

# Python
IF(NOT MR_EMSCRIPTEN)
IF(APPLE)
 include_directories(${HOMEBREW_PREFIX}/include)
ELSE()
 include_directories(/usr/include) 
ENDIF() # APPLE

set(PYBIND11_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/pybind11/include)
include_directories(${PYBIND11_DIR})
find_package(Python 3.9 EXACT REQUIRED)
find_package(PythonInterp 3.9 EXACT REQUIRED)
find_package(PythonLibs 3.9 EXACT REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})
ENDIF() # NOT MR_EMSCRIPTEN

set(PROJECT_SOURCE_DIR ./source)
include_directories(${PROJECT_SOURCE_DIR})

IF(MR_EMSCRIPTEN)
  add_compile_definitions(__emscripten__)
  link_directories(${THIRDPARTY_LIB_DIR}/lib)
  link_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/wasmtbb/build/linux_wasm32_emscripten_emscripten_wasm32_release)
  include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/wasmtbb/include)
  include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/spdlog/include)
  include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/jsoncpp/include)
  include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/googletest/googletest/include)
  include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/libzip/lib)
  include_directories(${THIRDPARTY_LIB_DIR}/libzip)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pthreads-mem-growth") # look https://github.com/emscripten-core/emscripten/issues/8287
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread -s PTHREAD_POOL_SIZE_STRICT=0 -s ALLOW_MEMORY_GROWTH=1 -s USE_LIBPNG=1")
ENDIF() # MR_EMSCRIPTEN

enable_testing()

IF(NOT MR_EMSCRIPTEN)
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
ENDIF() # NOT MR_EMSCRIPTEN

add_subdirectory(${PROJECT_SOURCE_DIR}/MRMesh ./MRMesh)
IF(NOT MR_EMSCRIPTEN)
  add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshpy ./mrmeshpy)
  add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshnumpy ./mrmeshnumpy)
  add_subdirectory(${PROJECT_SOURCE_DIR}/meshconv ./meshconv)
ENDIF() # NOT MR_EMSCRIPTEN
add_subdirectory(${PROJECT_SOURCE_DIR}/MRTest ./MRTest)
