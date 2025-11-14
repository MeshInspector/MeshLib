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

# Emscripten's Boost package doesn't contain CMake config files
# on other systems the module mode shows a warning
IF(NOT EMSCRIPTEN)
  find_package(Boost 1.73 CONFIG COMPONENTS headers REQUIRED)
ELSE()
  cmake_policy(PUSH)
  cmake_policy(SET CMP0167 OLD)
  find_package(Boost 1.73 REQUIRED)
  cmake_policy(POP)
ENDIF()
find_package(Eigen3 REQUIRED)
find_package(fmt 8 REQUIRED)
find_package(JsonCpp REQUIRED)
# libzip on Emscripten is unable to resolve its dependencies by itself
IF(EMSCRIPTEN)
  find_package(MbedTLS 1.0 REQUIRED)
  find_package(ZLIB REQUIRED)
ENDIF()
find_package(libzip REQUIRED)
find_package(spdlog 1.9 REQUIRED)
find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)
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
)

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

IF(NOT MRMESH_NO_GTEST)
  find_package(GTest REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE GTest::gtest)
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
