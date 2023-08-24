cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
# nvcc supoorts only c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if( ${MR_FEDORA} )
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2/")
    set(CMAKE_CUDA_PATH /usr/local/cuda-12.2)
else()
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1/")
    set(CMAKE_CUDA_PATH /usr/local/cuda-12.1)
endif()

set(CUDA_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}bin/nvcc)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}bin/nvcc)

find_package(CUDA 12 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std c++17 -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_52 \
            -gencode=arch=compute_52,code=sm_52 \
            -gencode=arch=compute_60,code=sm_60 \
            -gencode=arch=compute_61,code=sm_61 \
            -gencode=arch=compute_70,code=sm_70 \
            -gencode=arch=compute_75,code=sm_75 \
            -gencode=arch=compute_75,code=compute_75 ")
#enable_language(CUDA)
set(CUDA_VERBOSE_BUILD ON)

project(MRCuda CXX CUDA)

file(GLOB SOURCES "*.cpp" "*.cu")
file(GLOB HEADERS "*.h" "*.cuh")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE MRMesh )

set(CMAKE_INSTALL_PREFIX "/Library/Frameworks")
install(TARGETS ${PROJECT_NAME}
		EXPORT ${PROJECT_NAME}
		DESTINATION "${MR_MAIN_LIB_DIR}")

install(FILES ${HEADERS}
		DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
		DESTINATION ${MR_CONFIG_DIR}/cmake)


install(EXPORT ${PROJECT_NAME}
		FILE ${PROJECT_NAME}Targets.cmake
		NAMESPACE MeshLib::
		DESTINATION ${MR_CONFIG_DIR}/cmake)