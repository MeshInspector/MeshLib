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

IF(NOT MRMESH_NO_TIFF)
  find_package(TIFF REQUIRED)
  list(APPEND MRMESH_OPTIONAL_DEPENDENCIES TIFF::TIFF)
ENDIF()

IF(NOT MRMESH_NO_GTEST)
  IF(NOT EMSCRIPTEN)
    find_package(GTest REQUIRED)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES GTest::gtest)
  ELSE()
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES gtest)
  ENDIF()
ENDIF()

# alternative Eigen include for external project
find_package(Eigen3 NO_MODULE)
IF(DEFINED EIGEN3_FOUND)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIRS}>)
  list(APPEND MRMESH_OPTIONAL_DEPENDENCIES Eigen3::Eigen)
ENDIF()
IF(EMSCRIPTEN)
  # FIXME: comment required
  target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_STACK_ALLOCATION_LIMIT=0)
ENDIF()

IF(NOT EMSCRIPTEN)
  find_package(Boost 1.73 CONFIG COMPONENTS headers serialization REQUIRED)

  find_package(libzip QUIET)
  IF(NOT libzip_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(libzip REQUIRED IMPORTED_TARGET libzip)
    add_library(libzip::zip ALIAS PkgConfig::libzip)
  ENDIF()

  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      Boost::boost
      fmt::fmt
      JsonCpp::JsonCpp
      libzip::zip
      spdlog::spdlog
      TBB::tbb
    PRIVATE
      ${CMAKE_DL_LIBS}
      ${MRMESH_OPTIONAL_DEPENDENCIES}
  )

  IF(APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _GNU_SOURCE) # for Boost.Stacktrace

    find_library(COCOA_LIBRARY Cocoa ONLY)
    target_link_libraries(${PROJECT_NAME}
      PRIVATE
        ${COCOA_LIBRARY}
    )
  ELSE()
    # FIXME: incorrect thirdparty deployment on macOS
    find_package(tl-expected REQUIRED)
    target_link_libraries(${PROJECT_NAME}
      PUBLIC
        tl::expected
    )
  ENDIF()
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")

  target_compile_definitions(${PROJECT_NAME} PUBLIC SPDLOG_FMT_EXTERNAL)
  # hack to make it work with new version of fmt
  target_compile_definitions(${PROJECT_NAME} PUBLIC SPDLOG_WCHAR_FILENAMES)

  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      fmt
      jsoncpp
      spdlog
      tbb
      zip
    PRIVATE
      blosc
      mbedcrypto
      mbedtls
      mbedx509
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
