project(MRMesh CXX)

option(MRMESH_NO_TIFF "Disable TIFF support" OFF)
option(MRMESH_NO_GTEST "Disable building Google tests" OFF)

IF(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
  set(MRMESH_NO_GTEST ON)
ENDIF()

IF(MR_EMSCRIPTEN)
  set(MRMESH_NO_TIFF ON)
ENDIF()

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")
# MRPchCreate.cpp only drives the MSVC/MSBuild precompiled header; CMake generates its own.
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/MRPchCreate.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

# Emscripten's Boost package doesn't contain CMake config files
IF(EMSCRIPTEN)
  IF(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
  ENDIF()
  find_package(Boost 1.73 REQUIRED)
ELSE()
  find_package(Boost 1.73 CONFIG COMPONENTS headers REQUIRED)
ENDIF()
find_package(Eigen3 REQUIRED)
find_package(fmt 8 REQUIRED)
find_package(JsonCpp REQUIRED)
find_package(spdlog 1.9 REQUIRED)
find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)
find_package(tl-expected REQUIRED)
find_package(zlib-ng CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Boost::headers
    Eigen3::Eigen
    fmt::fmt
    JsonCpp::JsonCpp
    spdlog::spdlog
    TBB::tbb
    tl::expected
  PRIVATE
    ${CMAKE_DL_LIBS}
    zlib-ng::zlib
)

# libzip on Emscripten is unable to resolve its dependencies by itself
IF(EMSCRIPTEN)
  find_package(MbedTLS 1.0 REQUIRED)
ENDIF()
find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)
# libzip prior to version 1.11.4 has incorrect CMake config that requires libzip tools to be installed
IF(MESHLIB_USE_VCPKG OR APPLE OR EMSCRIPTEN)
  # vcpkg, homebrew, and submodule versions are already updated
  find_package(libzip REQUIRED)
ELSE()
  # on Ubuntu LTS (22.04, 24.04) we have to use pkg-config instead
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(libzip REQUIRED IMPORTED_TARGET libzip)
  add_library(libzip::zip ALIAS PkgConfig::libzip)
ENDIF()
target_link_libraries(${PROJECT_NAME} PRIVATE libzip::zip)

# TODO: CMake config
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${MESHLIB_THIRDPARTY_DIR}/parallel-hashmap>
)

IF(APPLE)
  # for Boost.Stacktrace
  target_compile_definitions(${PROJECT_NAME} PUBLIC _GNU_SOURCE)

  find_library(COCOA_LIBRARY Cocoa ONLY)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${COCOA_LIBRARY})
ENDIF()

IF(EMSCRIPTEN)
  target_compile_definitions(${PROJECT_NAME} PUBLIC
    EIGEN_STACK_ALLOCATION_LIMIT=0
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
ENDIF()

IF(NOT MRMESH_NO_TIFF)
  find_package(TIFF REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE TIFF::TIFF)
ENDIF()

IF(MR_PCH)
  IF(MSVC AND MR_PCH_USE_EXTRA_HEADERS)
    # MRMesh exports MRMESH_API, so it cannot reuse the shared (dllimport) PCH once that PCH
    # contains MeshLib headers; it builds its own from MRPch.h in its (dllexport) context.
    # MESHLIB_NO_VIEWER drops the viewer/imgui section of MRPch.h, which MRMesh (a lower layer)
    # neither needs nor has on its include path. See cmake/Modules/ConfigurePch.cmake.
    # Without the extra headers the shared PCH holds no MeshLib headers, so MRMesh just reuses it.
    target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VIEWER)
    target_precompile_headers(${PROJECT_NAME} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/../MRPch/MRPch.h>")
  ELSE()
    target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
  ENDIF()
ENDIF()

install(
  FILES ${HEADERS} "config_cmake.h"
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)
install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${MR_CONFIG_DIR}
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)
