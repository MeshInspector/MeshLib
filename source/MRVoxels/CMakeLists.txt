cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRVoxels CXX)

option(MRVOXELS_NO_DICOM "Disable DICOM format support" OFF)
option(MRVOXELS_NO_TIFF "Disable TIFF format support" OFF)

IF(MR_EMSCRIPTEN)
  set(MRVOXELS_NO_TIFF ON)
ENDIF()

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

IF(NOT MR_EMSCRIPTEN)
  find_package(fmt REQUIRED)
  find_package(spdlog REQUIRED)

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(jsoncpp REQUIRED QUIET IMPORTED_TARGET jsoncpp)
ENDIF()

set(MRVOXELS_OPTIONAL_DEPENDENCIES "")

IF(NOT MRVOXELS_NO_DICOM)
  find_package(GDCM CONFIG REQUIRED)
  list(APPEND MRVOXELS_OPTIONAL_DEPENDENCIES gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF)
ENDIF()

IF(NOT MRVOXELS_NO_TIFF)
  list(APPEND MRVOXELS_OPTIONAL_DEPENDENCIES tiff)
ENDIF()

IF(BUILD_TESTING)
  IF(APPLE)
    list(APPEND MRVOXELS_OPTIONAL_DEPENDENCIES GTest::gtest)
  ELSE()
    list(APPEND MRVOXELS_OPTIONAL_DEPENDENCIES gtest)
  ENDIF()
ENDIF()

IF(MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    openvdb
    jsoncpp
    spdlog
    tbb
    ${MRVOXELS_OPTIONAL_DEPENDENCIES}
  )
ELSEIF(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    OpenVDB
    PkgConfig::jsoncpp
    fmt
    spdlog
    tbb
    ${MRVOXELS_OPTIONAL_DEPENDENCIES}
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    openvdb
    fmt::fmt
    spdlog::spdlog
    PkgConfig::jsoncpp
    tbb
    ${MRVOXELS_OPTIONAL_DEPENDENCIES}
  )
ENDIF()

IF(WIN32 AND MINGW)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MRVOXELS_EXPORT)
ENDIF()

IF(MR_PCH)
  # TODO: compare performance
  #target_precompile_headers(${PROJECT_NAME} PRIVATE
  #  "$<$<COMPILE_LANGUAGE:CXX>:MRPch/MRPch.h>"
  #  "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MROpenVDB.h>"
  #)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
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
  FILES ${HEADERS}
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
