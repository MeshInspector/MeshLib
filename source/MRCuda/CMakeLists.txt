cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

IF(MESHLIB_EXPERIMENTAL_HIP)
  project(MRCuda CXX HIP)
ELSE()
  project(MRCuda CXX CUDA)
ENDIF()

include(CudaCompilerOptions)

option(MRCUDA_NO_VOXELS "Disable voxels support" OFF)

IF(NOT MESHLIB_BUILD_VOXELS)
  set(MRCUDA_NO_VOXELS ON)
ENDIF()

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")
file(GLOB CUDA_SOURCES "*.cu")
file(GLOB CUDA_HEADERS "*.cuh")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

set(MRCUDA_OPTIONAL_DEPENDENCIES "")

IF(NOT MRCUDA_NO_VOXELS)
  list(APPEND MRCUDA_OPTIONAL_DEPENDENCIES MRVoxels)
ENDIF()

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CUDA_INCLUDE_DIRS}
)

IF(WIN32 OR MESHLIB_USE_VCPKG)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    spdlog::spdlog
    fmt::fmt
    TBB::tbb
    ${MRCUDA_OPTIONAL_DEPENDENCIES}
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    spdlog
    fmt
    tbb
    ${MRCUDA_OPTIONAL_DEPENDENCIES}
  )
ENDIF()

IF(MESHLIB_EXPERIMENTAL_HIP)
  target_compile_options(${PROJECT_NAME} PRIVATE -Wno-unused-result)

  set_source_files_properties(${CUDA_SOURCES}
    PROPERTIES
    LANGUAGE HIP
  )

  add_library(${PROJECT_NAME}_device STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
  target_link_libraries(${PROJECT_NAME}_device PRIVATE hip::device)

  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      hip::host
    PRIVATE
      ${PROJECT_NAME}_device
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PUBLIC
    ${CUDART_LIBRARY}
  )
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

IF(MR_PCH)
  IF(CMAKE_CXX_STANDARD EQUAL MR_CXX_STANDARD)
    TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
    message( "MRCuda: using PCH" )
  ELSE()
    message( "MRCuda: not using PCH" )
  ENDIF()
ENDIF()
