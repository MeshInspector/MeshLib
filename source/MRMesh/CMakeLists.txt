cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

set(MRMESH_OPTIONAL_DEPENDENCIES "")
set(MRMESH_OPTIONAL_LIBRARY_DIRS "")
set(MRMESH_OPTIONAL_INCLUDE_DIRS "")

IF(NOT MRMESH_NO_TIFF)
  IF(WIN32 OR MESHLIB_USE_VCPKG)
    find_package(TIFF REQUIRED)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES TIFF::TIFF)
  ELSE()
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES tiff)
  ENDIF()
ENDIF()

IF(NOT MRMESH_NO_GTEST)
  IF(APPLE OR WIN32 OR MESHLIB_USE_VCPKG)
    find_package(GTest REQUIRED)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES GTest::gtest)
  ELSE()
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES gtest)
  ENDIF()
ENDIF()

IF(EMSCRIPTEN)
  # FIXME: comment required
  target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_STACK_ALLOCATION_LIMIT=0)
ENDIF()

IF(EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
ENDIF()

IF(EMSCRIPTEN)
  # hack to make it work with new version of fmt
  target_compile_definitions(${PROJECT_NAME} PUBLIC SPDLOG_WCHAR_FILENAMES)
ENDIF()

IF(NOT WIN32)
  # required for Boost.Stacktrace
  target_compile_definitions(${PROJECT_NAME} PRIVATE _GNU_SOURCE)
ENDIF()

IF(APPLE)
  find_library(COCOA_LIBRARY Cocoa ONLY)
  find_package(libzip REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    spdlog::spdlog
    libzip::zip
    Boost::boost
    JsonCpp::JsonCpp
    fmt::fmt
    TBB::tbb
    Eigen3::Eigen
    ${MRMESH_OPTIONAL_DEPENDENCIES}
    ${COCOA_LIBRARY}
    ${CMAKE_DL_LIBS}
  )
ELSEIF(EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    spdlog
    fmt
    blosc
    mbedtls
    mbedx509
    mbedcrypto
    zip
    jsoncpp
    tbb
    ${MRMESH_OPTIONAL_DEPENDENCIES}
  )
ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
  find_package(tl-expected REQUIRED)
  find_package(libzip REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC
    tl::expected
    fmt::fmt
    spdlog::spdlog
    libzip::zip
    Boost::boost
    JsonCpp::JsonCpp
    TBB::tbb
    Eigen3::Eigen
    ${CMAKE_DL_LIBS}
    ${MRMESH_OPTIONAL_DEPENDENCIES}
  )
ELSE()
  find_package(tl-expected REQUIRED)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    Eigen3::Eigen
    PRIVATE
    tl::expected
    fmt::fmt
    spdlog::spdlog
    zip
    Boost::boost
    JsonCpp::JsonCpp
    TBB::tbb
    ${CMAKE_DL_LIBS}
    ${MRMESH_OPTIONAL_DEPENDENCIES}
  )
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
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
