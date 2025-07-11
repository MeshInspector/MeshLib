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

set(MRVOXELS_OPTIONAL_DEPENDENCIES "")

IF(NOT MRVOXELS_NO_DICOM)
  find_package(GDCM CONFIG REQUIRED)
  list(APPEND MRVOXELS_OPTIONAL_DEPENDENCIES gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF)
ENDIF()

IF(NOT MRVOXELS_NO_TIFF)
  find_package(TIFF REQUIRED)
  list(APPEND MRVOXELS_OPTIONAL_DEPENDENCIES TIFF::TIFF)
ENDIF()

IF(EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    openvdb
    jsoncpp
    spdlog
    tbb
    ${MRVOXELS_OPTIONAL_DEPENDENCIES}
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    JsonCpp::JsonCpp
    fmt::fmt
    spdlog::spdlog
    TBB::tbb
    ${MRVOXELS_OPTIONAL_DEPENDENCIES}
  )

  IF(MESHLIB_USE_VCPKG)
    find_package(OpenVDB 10.1 COMPONENTS openvdb REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      OpenVDB::openvdb
    )
  ELSEIF(APPLE)
    # FIXME: OpenVDB's CMake config cannot be found
    target_link_libraries(${PROJECT_NAME} PRIVATE
      OpenVDB
    )
  ELSE()
    # TODO: upgrade to version 10.1 after dropping Ubuntu 20.04 support
    find_package(OpenVDB 9 COMPONENTS openvdb REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      OpenVDB::openvdb
    )
  ENDIF()
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

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
