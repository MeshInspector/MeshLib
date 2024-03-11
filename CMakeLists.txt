cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

IF(DEFINED ENV{MR_USE_CPP_23} AND "$ENV{MR_USE_CPP_23}" STREQUAL "ON")
  set(MR_CXX_STANDARD 23)
  message("enable c++23")
ELSE()
  set(MR_CXX_STANDARD 20)
ENDIF()
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
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
      message("Homebrew found. Prefix: ${HOMEBREW_PREFIX}")
    ELSE()
      message("Homebrew not found!")
      message(FATAL_ERROR "${CMD_ERROR} ${HOMEBREW_PREFIX}")
    ENDIF()
    # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
    link_directories(${HOMEBREW_PREFIX}/lib)
    set(CPPFLAGS "-I${HOMEBREW_PREFIX}/opt/llvm/include -I${HOMEBREW_PREFIX}/include")
    set(LDFLAGS "-L${HOMEBREW_PREFIX}/opt/llvm/lib -Wl,-rpath,${HOMEBREW_PREFIX}/opt/llvm/lib")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -framework Cocoa -framework IOKit") # https://github.com/pybind/pybind11/issues/382

    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib")
  ELSE()
    # uncomment to print compile time of each translation unit,
    # such print format is selected to simplify load in Excel with fields (user time, system time, command) subdivision by commas
    IF(EXISTS /usr/bin/time)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "/usr/bin/time -f \",%U,%S,%C\" -a -o compile_timings.txt")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "/usr/bin/time -f \",%U,%S,%C\" -a -o link_timings.txt")
    ENDIF()
  ENDIF() # APPLE
ENDIF() # NOT MR_EMSCRIPTEN

project(MeshLib CXX)
add_compile_definitions(MR_PROJECT_NAME=\"MeshLib\")
add_compile_definitions(MR_FRAMEWORK)
add_compile_definitions(IMGUI_USER_CONFIG=\"imgui/MRCustomImGuiConfig.h\")

# all binaries will be located in ./build/Release/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DDEBUG")
# turn on warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-sign-compare -Werror -fvisibility=hidden -pedantic-errors -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS -DIMGUI_ENABLE_FREETYPE")
IF(NOT MR_EMSCRIPTEN_SINGLETHREAD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD
IF(WIN32 AND MINGW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
ENDIF()
# make link to fail if there are unresolved symbols (GCC and Clang)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,defs")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs")
ENDIF()

option(MESHLIB_CUSTOM_INSTALL_PREFIX "Install into directory defined by CMAKE_INSTALL_PREFIX" OFF)
option(MESHLIB_PYTHON_SUPPORT "Python support" ON)
option(MESHLIB_BUILD_MRVIEWER "Build MRViewer library and application" ON)
option(MESHLIB_BUILD_PYTHON_MODULES "Build Python modules" ON)
option(MESHLIB_BUILD_MESHCONV "Build meshconv utility" ON)
option(MESHLIB_BUILD_MRCUDA "Build MRCuda library" ON)

include(CTest)

set(MESHLIB_THIRDPARTY_DIR "${PROJECT_SOURCE_DIR}/thirdparty")
set(MESHLIB_THIRDPARTY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(THIRDPARTY_LIB_DIR "${PROJECT_SOURCE_DIR}/lib")
set(MESHLIB_THIRDPARTY_LIB_DIR ${THIRDPARTY_LIB_DIR})
IF(NOT MESHLIB_CUSTOM_INSTALL_PREFIX)
  IF(NOT EXISTS "${THIRDPARTY_LIB_DIR}")
    message(FATAL_ERROR "thirdparty build directory not found! You can build thirdparty with ./scripts/build_thirdparty.sh")
  ENDIF()
  include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR})
  link_directories(${THIRDPARTY_LIB_DIR})
ENDIF()

find_package(PkgConfig REQUIRED)

set(MR_UBUNTU FALSE) # any Ubuntu version
set(MR_UBUNTU22 FALSE) # Ubuntu 22.*
set(MR_FEDORA FALSE)
set(MR_FEDORA39 FALSE)
set(MR_PLATFORM "UNKNOWN")

IF(NOT APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  add_compile_definitions(__ARM_CPU__)
  message("ARM cpu detected")
ENDIF()

IF(MR_EMSCRIPTEN)
  SET(MR_PLATFORM "WASM")
ELSE()
  IF(APPLE)
    include_directories(${HOMEBREW_PREFIX}/include)
    # need to hardcode some path to fix wheel (no option to pass path)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/..;@loader_path/../lib;@loader_path/../lib/lib;@loader_path/meshlib;${THIRDPARTY_LIB_DIR};${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(MR_PLATFORM "APPLE_${CMAKE_HOST_SYSTEM_PROCESSOR}")
    IF(BUILD_TESTING)
      set(GTest_DIR ${THIRDPARTY_LIB_DIR}/cmake/GTest)
      find_package(GTest REQUIRED CONFIG)
      include_directories(${GTEST_INCLUDE_DIRS})
    ENDIF()
  ELSE()
    # set platform dependent flags
    file(STRINGS /etc/os-release distro REGEX "^NAME=")
    string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" distro "${distro}")

    # check version (major)
    file(STRINGS /etc/os-release version_id REGEX "^VERSION_ID=")
    string(REGEX REPLACE "VERSION_ID=(.*)" "\\1" version_id "${version_id}")
    string(REGEX REPLACE "\"(.*)\"" "\\1" version_id "${version_id}")

    IF(${distro} STREQUAL "Ubuntu")
      string(FIND ${version_id} "." dot-pos)
      string(SUBSTRING ${version_id} 0 ${dot-pos} version_id)

      SET(MR_UBUNTU TRUE)
      IF(${version_id} STREQUAL "22")
        SET(MR_UBUNTU22 TRUE)
      ENDIF() # Ubuntu 22
    ENDIF() # Ubuntu

    IF(${distro} STREQUAL "Fedora Linux")
      SET(MR_FEDORA TRUE)
      IF(${version_id} STREQUAL "39")
        SET(MR_FEDORA39 TRUE)
      ENDIF() # Fedora 39
    ENDIF() # Fedora

    message("using ${distro} ${version_id}")
    set(MR_PLATFORM "${distro}_${version_id}")
  ENDIF() # APPLE
  add_definitions(-DMR_PLATFORM="${MR_PLATFORM}")
  message("platform: ${MR_PLATFORM}")
ENDIF() # NOT MR_EMSCRIPTEN

IF(NOT MR_EMSCRIPTEN)
  IF(MESHLIB_PYTHON_SUPPORT)
    IF(${MR_UBUNTU22} OR APPLE)
      set(MR_PYTHON_VERSION 3.10)
    ELSEIF(${MR_UBUNTU})
      set(MR_PYTHON_VERSION 3.8)
    ELSEIF(${MR_FEDORA39})
      set(MR_PYTHON_VERSION 3.12)
    ELSEIF(${MR_FEDORA})
      set(MR_PYTHON_VERSION 3.11)
    ELSE()
      set(MR_PYTHON_VERSION "")
    ENDIF()

    IF($ENV{MESHLIB_PYTHON_VERSION})
      set(MR_PYTHON_VERSION $ENV{MESHLIB_PYTHON_VERSION})
    ENDIF()

    IF(APPLE)
      link_directories(${HOMEBREW_PREFIX}/opt/python@${MR_PYTHON_VERSION}/Frameworks/Python.framework/Versions/${MR_PYTHON_VERSION}/lib/)
      include_directories(${HOMEBREW_PREFIX}/include)
      find_package(pybind11 REQUIRED)
    ENDIF()

    IF(NOT MR_PYTHON_VERSION STREQUAL "")
      set(MR_PYTHON_VERSION ${MR_PYTHON_VERSION} EXACT)
    ENDIF()

    find_package(Python ${MR_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)
    # legacy option, remove in the future
    set(MR_PYTHON_LIB Python::Python)
  ENDIF()
ENDIF()

set(PROJECT_SOURCE_DIR ./source)
include_directories(${PROJECT_SOURCE_DIR})

IF(MR_EMSCRIPTEN)
  IF(MR_EMSCRIPTEN_SINGLETHREAD)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web")
  ELSE()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web,worker -pthread -s PTHREAD_POOL_SIZE_STRICT=0 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pthreads-mem-growth") # look https://github.com/emscripten-core/emscripten/issues/8287
  ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_RUNTIME_METHODS=[ccall] -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB -s LLD_REPORT_UNDEFINED=1 -s USE_WEBGL2=1 -s USE_GLFW=3 -s USE_ZLIB=1 -s FULL_ES3=1 -s USE_LIBPNG=1")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --preload-file ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets@/")
  IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -sASYNCIFY -Wno-limited-postlink-optimizations")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -sASYNCIFY_IGNORE_INDIRECT -sASYNCIFY_ADD=@${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/wasm_async_func_list.txt")
    add_compile_definitions(MR_EMSCRIPTEN_ASYNCIFY)
  ENDIF() # NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY
ENDIF() # MR_EMSCRIPTEN

IF(NOT MR_EMSCRIPTEN)
  pkg_check_modules(JSONCPP jsoncpp)
  include_directories(${JSONCPP_INCLUDE_DIRS})

  find_package(fmt)
  find_package(spdlog REQUIRED)
  IF(spdlog_VERSION VERSION_GREATER_EQUAL "1.7" AND fmt_VERSION VERSION_GREATER_EQUAL "7.0")
    add_compile_definitions(SPDLOG_FMT_EXTERNAL)
  ENDIF()

  add_compile_definitions(SPDLOG_COMPILED_LIB SPDLOG_SHARED_LIB)
ENDIF() # NOT MR_EMSCRIPTEN

IF(NOT $ENV{MR_VERSION} STREQUAL "")
  set(MESHLIB_VERSION $ENV{MR_VERSION})
ELSE()
  set(MESHLIB_VERSION 0.0.0)
ENDIF()
message("version: ${MESHLIB_VERSION}")

IF(MESHLIB_CUSTOM_INSTALL_PREFIX)
  include(GNUInstallDirs)
  set(MR_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
  set(MR_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
  set(MR_MAIN_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
  set(MR_PY_LIB_DIR "${CMAKE_INSTALL_LIBDIR}/meshlib")
  set(MR_RESOURCES_DIR "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}")
  set(MR_FONTS_DIR "${CMAKE_INSTALL_DATAROOTDIR}/fonts")
  set(MR_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
ELSEIF(NOT APPLE)
  set(MR_ROOT "./usr/local")
  set(MR_BIN_DIR "${MR_ROOT}/bin")
  set(MR_INCLUDE_DIR "${MR_ROOT}/include/MeshLib")
  set(MR_MAIN_LIB_DIR "${MR_ROOT}/lib/MeshLib")
  set(MR_PY_LIB_DIR "${MR_MAIN_LIB_DIR}/meshlib")
  set(MR_RESOURCES_DIR "${MR_ROOT}/etc/MeshLib")
  set(MR_FONTS_DIR "${MR_ROOT}/share/fonts")
  set(MR_CONFIG_DIR "${MR_MAIN_LIB_DIR}/cmake")
ELSE()
  set(MR_ROOT "./Library/Frameworks/MeshLib.framework/Versions/${MESHLIB_VERSION}")
  set(MR_BIN_DIR "${MR_ROOT}/bin")
  set(MR_INCLUDE_DIR "${MR_ROOT}/include")
  set(MR_MAIN_LIB_DIR "${MR_ROOT}/lib")
  set(MR_PY_LIB_DIR "${MR_MAIN_LIB_DIR}/meshlib")
  set(MR_RESOURCES_DIR "${MR_ROOT}/Resources")
  set(MR_FONTS_DIR "${MR_ROOT}/Resources/fonts")
  set(MR_CONFIG_DIR "${MR_ROOT}/Resources/cmake")
ENDIF()

# MRMesh library is always built
add_subdirectory(${PROJECT_SOURCE_DIR}/MRMesh ./MRMesh)
add_subdirectory(${PROJECT_SOURCE_DIR}/MRPch ./MRPch)

IF(MESHLIB_BUILD_MRVIEWER)
  set(IMGUI_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/imgui)
  include_directories(${IMGUI_DIR})

  IF(NOT MR_EMSCRIPTEN)
    find_package(glfw3 CONFIG REQUIRED)
  ENDIF()

  add_subdirectory(${PROJECT_SOURCE_DIR}/MRViewer ./MRViewer)
  add_subdirectory(${PROJECT_SOURCE_DIR}/MRCommonPlugins ./MRCommonPlugins)
  add_subdirectory(${PROJECT_SOURCE_DIR}/MRViewerApp ./MRViewerApp)
ENDIF()

IF(NOT MESHLIB_PYTHON_SUPPORT)
  set(MESHLIB_BUILD_PYTHON_MODULES OFF)
ENDIF()
IF(NOT MR_EMSCRIPTEN)
  IF(MESHLIB_BUILD_PYTHON_MODULES)
    add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshpy ./mrmeshpy)
    add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshnumpy ./mrmeshnumpy)
    IF(MESHLIB_BUILD_MRVIEWER)
      add_subdirectory(${PROJECT_SOURCE_DIR}/mrviewerpy ./mrviewerpy)
    ENDIF()
  ENDIF()
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  IF(MESHLIB_BUILD_MESHCONV)
    add_subdirectory(${PROJECT_SOURCE_DIR}/meshconv ./meshconv)
  ENDIF()
ENDIF()

IF(NOT MR_EMSCRIPTEN AND NOT APPLE)
  IF(MESHLIB_BUILD_MRCUDA)
    add_subdirectory(${PROJECT_SOURCE_DIR}/MRCuda ./MRCuda)
  ENDIF()
ENDIF()

IF(BUILD_TESTING)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/MRTest ./MRTest)
ENDIF()

include(CMakePackageConfigHelpers)
configure_package_config_file(meshlib-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/meshlib-config.cmake
  INSTALL_DESTINATION ${MR_CONFIG_DIR}
  PATH_VARS MR_INCLUDE_DIR MR_MAIN_LIB_DIR
)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/meshlib-config-version.cmake
  VERSION ${MESHLIB_VERSION}
  COMPATIBILITY ExactVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/meshlib-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/meshlib-config-version.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

set(CPACK_GENERATOR "DRAGNDROP")
include(CPack)
