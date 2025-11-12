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

IF(MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    openvdb
    jsoncpp
    spdlog
    tbb
  )
ELSE()
  # TODO: correct OpenVDB module search
  IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND CMAKE_MODULE_PATH
      "/usr/lib/cmake/OpenVDB/"
      "/usr/lib/x86_64-linux-gnu/cmake/OpenVDB/"
      "/usr/lib/aarch64-linux-gnu/cmake/OpenVDB/"
      "/usr/lib64/cmake/OpenVDB/"
    )
  ELSEIF(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(APPEND CMAKE_MODULE_PATH
      "${HOMEBREW_PREFIX}/lib/cmake/OpenVDB/"
    )
  ENDIF()
  find_package(OpenVDB REQUIRED)

  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      MRMesh
      OpenVDB::openvdb
  )
ENDIF()

IF(NOT MRVOXELS_NO_DICOM)
  find_package(GDCM CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF)
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
