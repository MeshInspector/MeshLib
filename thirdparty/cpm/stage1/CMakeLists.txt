# to force update thirdparty touch this line
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(${CMAKE_CURRENT_LIST_DIR}/../common_pre_project.cmake)
project(MeshLib_thirdparty_stage1 CXX)
include(${CMAKE_CURRENT_LIST_DIR}/../common_post_project.cmake)

# Libraries with no dependency on anything else built here. Everything that reaches for a
# sibling through find_package lives in ../stage2, which is configured after this one is
# installed.

IF(EMSCRIPTEN)
  set(BOOST_INCLUDE_LIBRARIES locale)
  set(BOOST_LOCALE_ENABLE_ICONV OFF)
  set(BOOST_SKIP_INSTALL_RULES OFF)
  meshlib_add_package(boost-libs)

  set(WITH_JAVA OFF)
  set(WITH_JPEG8 ON)
  set(WITH_SIMD OFF)
  set(BUILD_TESTING OFF)
  set(ENABLE_STATIC OFF)
  set(CMAKE_INSTALL_DOCDIR_BACKUP "${CMAKE_INSTALL_DOCDIR}")
  set(CMAKE_INSTALL_MANDIR_BACKUP "${CMAKE_INSTALL_MANDIR}")
  set(CMAKE_INSTALL_DOCDIR "share/doc/libjpeg-turbo")
  set(CMAKE_INSTALL_MANDIR "share/man")
  meshlib_add_package(libjpeg-turbo)
  set(CMAKE_INSTALL_DOCDIR "${CMAKE_INSTALL_DOCDIR_BACKUP}")
  set(CMAKE_INSTALL_MANDIR "${CMAKE_INSTALL_MANDIR_BACKUP}")

  set(ENABLE_PROGRAMS OFF CACHE BOOL "")
  set(ENABLE_TESTING OFF CACHE BOOL "")
  set(DISABLE_PACKAGE_CONFIG_AND_INSTALL OFF CACHE BOOL "")
  meshlib_add_package(mbedtls)

  set(INSTALL_GTEST ON)
  add_compile_definitions(GTEST_HAS_CXXABI_H_=0)
  meshlib_add_package(googletest)
ENDIF()

IF(NOT APPLE AND NOT MR_EMSCRIPTEN)
  set(BUILD_SHARED_LIBS ON)

  find_package(cpr)
  IF(NOT cpr_FOUND)
    set(CPR_ENABLE_UNINSTALL "OFF")
    set(CPR_ENABLE_INSTALL "OFF")
    set(CPR_USE_SYSTEM_GTEST "ON")
    set(CPR_BUILD_TESTS "OFF")
    set(CPR_BUILD_TESTS_SSL "OFF")
    set(CPR_FORCE_USE_SYSTEM_CURL ON)
    meshlib_add_package(cpr)
  ENDIF()

  find_package(nlohmann_json)
  if(NOT TARGET nlohmann_json::nlohmann_json)
    set(JSON_BuildTests OFF)
    set(JSON_Install ON)
    meshlib_add_package(nlohmann-json)
  endif()

  find_package(httplib)
  if(NOT TARGET httplib::httplib)
    set(HTTPLIB_TEST OFF)
    set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF)
    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF)
    set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF)
    set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF)
    meshlib_add_package(cpp-httplib)
  endif()

  find_package(PkgConfig)
  IF(PkgConfig_FOUND)
    pkg_check_modules(tinyxml2 QUIET tinyxml2)
  ENDIF()
  IF(NOT tinyxml2_FOUND OR tinyxml2_VERSION VERSION_LESS 10)
    meshlib_add_package(tinyxml2)
  ENDIF()
ENDIF()

meshlib_add_package(OpenCTM)

IF(NOT APPLE)
  set(ZLIB_COMPAT       OFF CACHE BOOL "")
  set(ZLIB_ENABLE_TESTS OFF CACHE BOOL "")
  set(ZLIBNG_ENABLE_TESTS OFF CACHE BOOL "")
  set(WITH_GTEST        OFF CACHE BOOL "")
  set(WITH_GZFILEOP     OFF CACHE BOOL "")
  IF(EMSCRIPTEN)
    # Force static: undefined BUILD_SHARED_LIBS makes zlib-ng emit duplicate
    # libz-ng.a rules under Emscripten. Restore to undefined (not empty)
    # afterwards -- jsoncpp downstream breaks on empty BUILD_SHARED_LIBS.
    set(BUILD_SHARED_LIBS OFF)
    meshlib_add_package(zlib-ng)
    unset(BUILD_SHARED_LIBS)
  ELSE()
    meshlib_add_package(zlib-ng)
  ENDIF()

  option(PHMAP_INSTALL "" ON)
  meshlib_add_package(parallel-hashmap)
ENDIF()

IF(EMSCRIPTEN)
  meshlib_add_package(expected)

  set(GDCM_BUILD_APPLICATIONS OFF)
  set(GDCM_BUILD_SHARED_LIBS OFF)
  set(GDCM_BUILD_TESTING OFF)
  set(GDCM_BUILD_DOCBOOK_MANPAGES OFF)
  meshlib_add_package(GDCM)

  set(JSONCPP_WITH_TESTS OFF)
  set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF)
  set(BUILD_STATIC_LIBS OFF)
  meshlib_add_package(jsoncpp)

  set(FMT_INSTALL ON)
  set(FMT_MODULE OFF)
  set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
  meshlib_add_package(fmt)

  set(SPDLOG_INSTALL ON)
  set(SPDLOG_FMT_EXTERNAL ON)
  meshlib_add_package(spdlog)

  option(TBB_TEST "Enable testing" OFF)
  option(TBBMALLOC_BUILD "Enable tbbmalloc build" OFF)
  set(CMAKE_SYSTEM_PROCESSOR_BACKUP "${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_SYSTEM_PROCESSOR "unknown")
  meshlib_add_package(onetbb)
  set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR_BACKUP}")

  # oneTBB forces -fexceptions under Emscripten
  get_target_property(tbb_compile_options tbb COMPILE_OPTIONS)
  list(REMOVE_ITEM tbb_compile_options -fexceptions)
  set_target_properties(tbb PROPERTIES COMPILE_OPTIONS "${tbb_compile_options}")
  install(DIRECTORY ${onetbb_SOURCE_DIR}/include/tbb DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR})

  meshlib_add_package(tinyxml2)

  set(BUILD_TESTS OFF CACHE BOOL "")
  set(BUILD_FUZZERS OFF CACHE BOOL "")
  set(BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(BUILD_STATIC OFF CACHE BOOL "")
  set(BLOSC_INSTALL ON)
  meshlib_add_package(c-blosc)

  set(EIGEN_BUILD_DOC OFF)
  set(BUILD_TESTING OFF)
  set(EIGEN_BUILD_CMAKE_PACKAGE ON)
  set(EIGEN_BUILD_PKGCONFIG ON)
  meshlib_add_package(eigen)
ELSE() # not MR_EMSCRIPTEN
  set(BUILD_SHARED_LIBS ON)

  set(CLIP_EXAMPLES OFF CACHE BOOL "")
  set(CLIP_TESTS OFF CACHE BOOL "")
  set(CLIP_X11_WITH_PNG OFF CACHE BOOL "")
  meshlib_add_package(clip)
ENDIF()

find_package(glad)
IF(NOT glad_FOUND)
  set(GLAD_INSTALL ON CACHE BOOL "")
  set(GLAD_REPRODUCIBLE ON CACHE BOOL "")
  meshlib_add_package(glad)
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  set(PYBIND11_INSTALL ON)
  option(PYBIND11_NONLIMITEDAPI_BUILD_STUBS "" ON)
  option(PYBIND11_NONLIMITEDAPI_INSTALL_EXPORTS "" ON)
  set(PYBIND11_NONLIMITEDAPI_SUFFIX "meshlib" CACHE STRING "")
  set(PYBIND11_NONLIMITEDAPI_PYTHON_MIN_VERSION_HEX "0x030800f0" CACHE STRING "") # Python 3.8
  set(PYBIND11_NONLIMITEDAPI_INTERNALS_VERSION "5" CACHE STRING "")
  set(PYBIND11_NONLIMITEDAPI_COMPILER_TYPE_STRING "_meshlib" CACHE STRING "")
  set(PYBIND11_NONLIMITEDAPI_BUILD_ABI_STRING "_meshlib" CACHE STRING "")
  IF(APPLE)
    set(PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION "3.10" CACHE STRING "")
    execute_process(
      COMMAND brew --prefix
      OUTPUT_VARIABLE HOMEBREW_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(Python_ROOT_DIR "${HOMEBREW_PREFIX}/opt/python@${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION}")
  ENDIF()
  meshlib_add_package(mrbind-pybind11)
ENDIF()

set(TINYGLTF_INSTALL ON)
meshlib_add_package(tinygltf)

set(WITH_TESTS OFF CACHE BOOL "")
set(WITH_EMBIND OFF CACHE BOOL "")
meshlib_add_package(laz-perf)

set(DRACO_JS_GLUE OFF CACHE BOOL "")
set(DRACO_TESTS OFF CACHE BOOL "")
IF(EMSCRIPTEN)
  # Draco checks these even with its JS glue disabled
  set(PYTHONINTERP_FOUND ON)
  set(ENV{EMSCRIPTEN} "${EMSCRIPTEN_ROOT_PATH}")
ENDIF()
meshlib_add_package(draco)
