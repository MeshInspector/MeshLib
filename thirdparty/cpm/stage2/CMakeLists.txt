# to force update thirdparty touch this line
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(${CMAKE_CURRENT_LIST_DIR}/../common_pre_project.cmake)
project(MeshLib_thirdparty_stage2 CXX)
include(${CMAKE_CURRENT_LIST_DIR}/../common_post_project.cmake)

# Libraries that resolve a dependency built by ../stage1 through find_package, so they can only
# be configured once stage1 has been installed into CMAKE_INSTALL_PREFIX:
#   libzip -> MbedTLS, libE57Format -> tinyxml2, openvdb -> TBB + Blosc + Boost, fastmcpp ->
#   httplib + nlohmann_json.

IF(NOT APPLE AND NOT MR_EMSCRIPTEN)
  find_package(OpenVDB 10 COMPONENTS openvdb)
  IF(NOT OpenVDB_FOUND)
    set(OPENVDB_ENABLE_UNINSTALL OFF)
    set(OPENVDB_ENABLE_INSTALL OFF)
    set(OPENVDB_CORE_SHARED "ON")
    set(OPENVDB_CORE_STATIC "OFF")
    set(OPENVDB_BUILD_BINARIES OFF)
    set(OPENVDB_BUILD_VDB_PRINT OFF)
    set(USE_EXPLICIT_INSTANTIATION OFF)
    meshlib_add_package(openvdb)
  ENDIF()
ENDIF()

IF(EMSCRIPTEN)
  set(BUILD_TOOLS OFF CACHE BOOL "")
  set(BUILD_REGRESS OFF CACHE BOOL "")
  set(BUILD_EXAMPLES OFF CACHE BOOL "")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s USE_ZLIB=1")
  set(LIBZIP_DO_INSTALL ON)
  meshlib_add_package(libzip)

  set(E57_BUILD_TEST OFF)
  set(E57_XML_PARSER TinyXML2)
  meshlib_add_package(libE57Format)

  set(OPENVDB_ENABLE_UNINSTALL OFF)
  set(OPENVDB_ENABLE_INSTALL OFF)
  set(OPENVDB_CORE_SHARED "ON")
  set(OPENVDB_CORE_STATIC "OFF")
  set(OPENVDB_BUILD_BINARIES OFF)
  set(OPENVDB_BUILD_VDB_PRINT OFF)
  set(OPENVDB_USE_DELAYED_LOADING OFF)
  set(USE_EXPLICIT_INSTANTIATION OFF)
  set(Tbb_VERSION 2021.12)
  set(CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-template-arg-list-after-template-kw")
  meshlib_add_package(openvdb)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BACKUP}")
ELSE() # not MR_EMSCRIPTEN
  set(BUILD_SHARED_LIBS ON)

  set(E57_GIT_SUBMODULE_UPDATE OFF)
  set(E57_XML_PARSER TinyXML2 CACHE STRING "")
  meshlib_add_package(libE57Format)

  set(FASTMCPP_BUILD_TESTS OFF)
  set(FASTMCPP_BUILD_EXAMPLES OFF)
  set(FASTMCPP_FETCH_CURL OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  meshlib_add_package(fastmcpp)
ENDIF()
