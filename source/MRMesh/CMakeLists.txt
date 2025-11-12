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
  find_package(GTest REQUIRED)
  list(APPEND MRMESH_OPTIONAL_DEPENDENCIES GTest::gtest)
ENDIF()

IF(EMSCRIPTEN)
  target_compile_definitions(${PROJECT_NAME} PUBLIC
    EIGEN_STACK_ALLOCATION_LIMIT=0
    SPDLOG_FMT_EXTERNAL
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")

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
ELSE()
  find_package(libzip REQUIRED)
  find_package(tl-expected REQUIRED)
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
      libzip::zip
      ${CMAKE_DL_LIBS}
      ${MRMESH_OPTIONAL_DEPENDENCIES}
  )

  IF(APPLE)
    # for Boost.Stacktrace
    target_compile_definitions(${PROJECT_NAME} PUBLIC _GNU_SOURCE)

    find_library(COCOA_LIBRARY Cocoa ONLY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${COCOA_LIBRARY})
  ENDIF()
ENDIF()

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
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
