cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
# nvcc supports only c++20 and Cmake 3.16 from Ubuntu 20 does not support set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(CMAKE_VERSION VERSION_LESS 3.25.2)
  # more info: https://gitlab.kitware.com/cmake/cmake/-/issues/23079
  IF(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
    set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
  ENDIF()
ENDIF()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
set(CMAKE_CUDA_PATH /usr/local/cuda/)
set(CUDA_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

# without it we get "nvcc warning : The -std=c++20 flag is not supported with the configured host compiler. Flag will be ignored."
# https://stackoverflow.com/q/77170793/7325599
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
message("CMAKE_CXX_COMPILER=CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")

# TODO: migrate to FindCUDAToolkit (CMake 3.17 or later)
find_package(CUDA 12 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
  --std c++20 \
  -use_fast_math \
  -arch=sm_52 \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_75,code=sm_75 \
  -gencode=arch=compute_75,code=compute_75"
)
set(CUDA_VERBOSE_BUILD ON)

project(MRCuda CXX CUDA)

option(MRCUDA_NO_VOXELS "Disable voxels support" OFF)

IF(NOT MESHLIB_BUILD_VOXELS)
  set(MRCUDA_NO_VOXELS ON)
ENDIF()

file(GLOB SOURCES "*.cpp" "*.cu")
file(GLOB HEADERS "*.h" "*.hpp" "*.cuh")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

set(MRCUDA_OPTIONAL_DEPENDENCIES "")

IF(NOT MRCUDA_NO_VOXELS)
  list(APPEND MRCUDA_OPTIONAL_DEPENDENCIES MRVoxels)
ENDIF()

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  spdlog
  fmt
  tbb
  ${MRCUDA_OPTIONAL_DEPENDENCIES}
)

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
