cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(NOT MR_EMSCRIPTEN)
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
 set(CPPFLAGS "-I${HOMEBREW_PREFIX}/opt/llvm/include -I${HOMEBREW_PREFIX}/include")
 set(LDFLAGS "-L${HOMEBREW_PREFIX}/opt/llvm/lib -Wl,-rpath,${HOMEBREW_PREFIX}/opt/llvm/lib")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++20-stdlib=libc++ -undefined dynamic_lookup -framework Cocoa -framework IOKit") # https://github.com/pybind/pybind11/issues/382
ELSE()
 # uncomment to print compile time of each translation unit,
 # such print format is selected to simplify load in Excel with fields (user time, system time, command) subdivision by commas
 set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "/usr/bin/time -f \",%U,%S,%C\" -a -o compile_timings.txt")
 set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "/usr/bin/time -f \",%U,%S,%C\" -a -o link_timings.txt")
ENDIF() # APPLE
ENDIF() # NOT MR_EMSCRIPTEN

project (MeshLib CXX)
add_compile_definitions(MR_PROJECT_NAME=\"MeshViewer\")

# all binaries will be located in ./build/Release/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DDEBUG")
# turn on warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wno-sign-compare -Werror -fvisibility=hidden -pedantic-errors -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS -DIMGUI_ENABLE_FREETYPE")
IF(NOT MR_EMSCRIPTEN_SINGLETHREAD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD

set(MESHLIB_THIRDPARTY_DIR "${PROJECT_SOURCE_DIR}/thirdparty")
set(MESHLIB_THIRDPARTY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(THIRDPARTY_LIB_DIR "${PROJECT_SOURCE_DIR}/lib")
set(MESHLIB_THIRDPARTY_LIB_DIR ${THIRDPARTY_LIB_DIR})

IF(NOT EXISTS "${THIRDPARTY_LIB_DIR}")
 message(FATAL_ERROR "thirdparty build directory not found! You can build thirdparty with ./scripts/build_thirdparty.sh")
ENDIF()
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR})
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/imgui)
link_directories(${THIRDPARTY_LIB_DIR})

find_package(PkgConfig REQUIRED)

SET(MR_UBUNTU FALSE) # any Ubuntu version
SET(MR_UBUNTU22 FALSE) # Ubuntu 22.*
SET(MR_FEDORA FALSE)
# Python
IF(NOT MR_EMSCRIPTEN)
IF(APPLE)
 include_directories(${HOMEBREW_PREFIX}/include)
  # need to hardcode some path to fix wheel (no option to pass path)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../;@loader_path/meshlib;${THIRDPARTY_LIB_DIR};${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
ELSE()
 # set platform dependent flags
 file(STRINGS /etc/os-release distro REGEX "^NAME=")
 string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" distro "${distro}")

 # check version (major)
 file(STRINGS /etc/os-release version_id REGEX "^VERSION_ID=")
 string(REGEX REPLACE "VERSION_ID=\"(.*)\"" "\\1" version_id "${version_id}")
 string(FIND ${version_id} "." dot-pos)
 string(SUBSTRING ${version_id} 0 ${dot-pos} version_id)
 message("using ${distro} ${version_id}")

 IF( ${distro} STREQUAL "Ubuntu")
  SET(MR_UBUNTU TRUE)
  IF( ${version_id} STREQUAL "22")
   SET(MR_UBUNTU22 TRUE)
  ENDIF() # Ubuntu 22
 ENDIF() # Ubuntu

 IF( ${distro} STREQUAL "Fedora Linux")
  SET(MR_FEDORA TRUE)
 ENDIF() # Fedora
ENDIF() # APPLE

IF(${MR_UBUNTU22} OR APPLE)
 SET(PYTHON_VERSION 3.10)
ELSEIF(${MR_UBUNTU})
 SET(PYTHON_VERSION 3.8)
ELSE()
 SET(PYTHON_VERSION 3.9)
ENDIF()
IF ($ENV{MESHLIB_PYTHON_VERSION})
 SET(PYTHON_VERSION $ENV{MESHLIB_PYTHON_VERSION})
ENDIF()

IF(APPLE)
 LINK_DIRECTORIES(${HOMEBREW_PREFIX}/opt/python@${PYTHON_VERSION}/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib/)
ENDIF()

set(PYTHON_VAR "python")
string(CONCAT PYTHON_LIB ${PYTHON_VAR}${PYTHON_VERSION})

find_package(Python ${PYTHON_VERSION} EXACT REQUIRED)
find_package(PythonInterp ${PYTHON_VERSION} EXACT REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION} EXACT REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})
ENDIF() # NOT MR_EMSCRIPTEN

IF(APPLE)
 include_directories(${HOMEBREW_PREFIX}/include)
 find_package(pybind11 REQUIRED)
ENDIF()

set(PROJECT_SOURCE_DIR ./source)
include_directories(${PROJECT_SOURCE_DIR})

IF(MR_EMSCRIPTEN)
  add_compile_definitions(__emscripten__)
  link_directories(${MESHLIB_THIRDPARTY_DIR}/wasmtbb/build/linux_wasm32_emscripten_emscripten_wasm32_release)
  IF(NOT MR_EMSCRIPTEN_SINGLETHREAD)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread -s PTHREAD_POOL_SIZE_STRICT=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pthreads-mem-growth") # look https://github.com/emscripten-core/emscripten/issues/8287
  ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_RUNTIME_METHODS=[ccall] -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency -s ALLOW_MEMORY_GROWTH=1 -s LLD_REPORT_UNDEFINED=1 -s USE_WEBGL2=1 -s USE_GLFW=3 -s USE_ZLIB=1 -s FULL_ES3=1 -s USE_LIBPNG=1")
ENDIF() # MR_EMSCRIPTEN

enable_testing()

IF(NOT MR_EMSCRIPTEN)

find_package(glfw3 CONFIG REQUIRED)
pkg_check_modules(GTKMM gtkmm-3.0)
pkg_check_modules(JSONCPP jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})

IF(NOT APPLE)
 IF(${MR_FEDORA} OR ${MR_UBUNTU22})
  add_compile_definitions(SPDLOG_FMT_EXTERNAL)
 ENDIF()
ENDIF()
add_compile_definitions(SPDLOG_COMPILED_LIB SPDLOG_SHARED_LIB)
ENDIF() # NOT MR_EMSCRIPTEN

add_subdirectory(${PROJECT_SOURCE_DIR}/MRMesh ./MRMesh)
add_subdirectory(${PROJECT_SOURCE_DIR}/MRViewer ./MRViewer)
add_subdirectory(${PROJECT_SOURCE_DIR}/MRCommonPlugins ./MRCommonPlugins)
IF(NOT MR_EMSCRIPTEN)
  add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshpy ./mrmeshpy)
  add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshnumpy ./mrmeshnumpy)
  add_subdirectory(${PROJECT_SOURCE_DIR}/mrviewerpy ./mrviewerpy)
  add_subdirectory(${PROJECT_SOURCE_DIR}/meshconv ./meshconv)
ENDIF() # NOT MR_EMSCRIPTEN
add_subdirectory(${PROJECT_SOURCE_DIR}/MRTest ./MRTest)
add_subdirectory(${PROJECT_SOURCE_DIR}/MRViewerApp ./MRViewerApp)
