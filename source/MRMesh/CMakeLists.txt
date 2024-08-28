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

IF(NOT MR_EMSCRIPTEN)
  IF(NOT MRMESH_NO_TIFF)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES tiff)
  ENDIF()

  IF(NOT MRMESH_NO_GTEST)
    IF(APPLE)
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES GTest::gtest)
    ELSE()
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES gtest)
    ENDIF()
  ENDIF()
ENDIF()

# alternative Eigen include for external project
find_package(Eigen3 NO_MODULE)
IF(DEFINED EIGEN3_FOUND)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIRS}>)
  list(APPEND MRMESH_OPTIONAL_DEPENDENCIES Eigen3::Eigen)
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  find_package(Boost COMPONENTS REQUIRED)
  find_package(TBB REQUIRED)
  pkg_check_modules(JSONCPP jsoncpp)
  include_directories(${JSONCPP_INCLUDE_DIRS})
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
ENDIF()

IF(APPLE OR MR_EMSCRIPTEN)
  #jpeg-turbo is keg-only, which means it was not symlinked into /opt/homebrew,
  #because libjpeg-turbo is not linked to prevent conflicts with the standard libjpeg.
  #include_directories(/opt/homebrew/opt/jpeg-turbo/include)
  #link_directories(/opt/homebrew/opt/jpeg-turbo/lib)

  IF(APPLE)
    find_library(COCOA_LIBRARY Cocoa ONLY)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _GNU_SOURCE)
    target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_INCLUDE_DIRS}
    )
    target_link_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_LIBRARY_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
      spdlog
      zip
      Boost::boost
      jsoncpp
      fmt
      tbb
      ${MRMESH_OPTIONAL_DEPENDENCIES}
      ${COCOA_LIBRARY}
    )
  ELSE()
    target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_INCLUDE_DIRS}
    )
    target_link_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_LIBRARY_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
      blosc
      mbedtls
      mbedx509
      mbedcrypto
      zip
      jsoncpp
      tbb
      ${MRMESH_OPTIONAL_DEPENDENCIES}
    )
  ENDIF()
ELSE()
  find_package(tl-expected REQUIRED)
  target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
    ${MRMESH_OPTIONAL_INCLUDE_DIRS}
  )
  target_link_directories(${PROJECT_NAME} BEFORE PRIVATE
    ${MRMESH_OPTIONAL_LIBRARY_DIRS}
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE
    tl::expected
    fmt
    spdlog
    zip
    Boost::boost
    dl
    jsoncpp
    tbb
    ${MRMESH_OPTIONAL_DEPENDENCIES}
  )
ENDIF()

IF(WIN32 AND MINGW)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_EXPORT)
ENDIF()

set(CMAKE_INSTALL_PREFIX "/Library/Frameworks")
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)

install(
  FILES ${HEADERS} "config_cmake.h"
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
