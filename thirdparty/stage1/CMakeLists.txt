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
    set(tinyxml2_SHARED_LIBS ON)
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

  set(CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s NO_DISABLE_EXCEPTION_CATCHING=1")
  set(JSONCPP_WITH_TESTS OFF)
  set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF)
  set(BUILD_STATIC_LIBS OFF)
  meshlib_add_package(jsoncpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BACKUP}")

  meshlib_add_package(fmt)

  set(SPDLOG_INSTALL ON)
  set(SPDLOG_FMT_EXTERNAL ON)
  meshlib_add_package(spdlog)

  option(TBB_TEST "Enable testing" OFF)
  set(CMAKE_SYSTEM_PROCESSOR_BACKUP "${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_SYSTEM_PROCESSOR "unknown")
  meshlib_add_package(onetbb)
  set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR_BACKUP}")
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
  # We want to build the stubs in our main `CMakeLists.txt` instead of here.
  # It seems to work fine either way, but I'm not comfortable not having the stubs see the customized cmake variables
  #   that we set in our main `CMakeLists.txt`, such as those influencing the Pybind ABI strings.
  # Even though the stubs shouldn't call any Pybind functions right now, they still include the Pybind header.
  # Accidentally calling some inline Pybind function in it could have weird effects, I reckon.
  # Note that regardless of this setting, the python-version-specific shims always need to be built in our main `CMakeLists.txt`,
  #   to see our configuration CMake variables.
  option(PYBIND11_NONLIMITEDAPI_BUILD_STUBS "" OFF)
  add_subdirectory(${MESHLIB_THIRDPARTY_DIR}/mrbind-pybind11 ./mrbind-pybind11)
ENDIF()

set(TINYGLTF_INSTALL ON)
meshlib_add_package(tinygltf)

set(WITH_TESTS OFF CACHE BOOL "")
set(WITH_EMBIND OFF CACHE BOOL "")
meshlib_add_package(laz-perf)
