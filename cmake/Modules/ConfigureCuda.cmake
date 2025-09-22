IF(NOT CUDAToolkit_FOUND AND NOT CUDA_FOUND)
  IF(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    IF(WIN32)
      # For some reason setting C++20 on Windows appears to be ignored, so that even
      #   the single-argument version of `static_assert` doesn't compile (which is a C++17 feature).
      # Setting C++17 does work though.
      set(CMAKE_CUDA_STANDARD 17)

      # Right now we only seem have the old Cuda 11 in VS 2019 in CI.
      find_package(CUDAToolkit 11 REQUIRED)

      # For our VS2022 CI:
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
    ELSE()
      set(CMAKE_CUDA_STANDARD 20)
      find_package(CUDAToolkit 12 REQUIRED)
    ENDIF()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    IF(NOT DEFINED CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    ENDIF()

    IF(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
      option(MRCUDA_OVERRIDE_HOST_COMPILER "Override the default nvcc host compiler with the one used for C++ compilation" OFF)
      IF(MRCUDA_OVERRIDE_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
      ENDIF()
    ENDIF()

    IF(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      IF(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 89-real 120)
      ELSEIF(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES 52-real 60-real 61-real 70-real 75-real 86-real 89)
      ELSE()
        set(CMAKE_CUDA_ARCHITECTURES 52-real 60-real 61-real 70-real 75)
      ENDIF()
    ENDIF()

    set(CUDART_LIBRARY CUDA::cudart_static)
  ELSE()
    # nvcc supports only c++20 and Cmake 3.16 from Ubuntu 20 does not support set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # more info: https://gitlab.kitware.com/cmake/cmake/-/issues/23079
    IF(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
      set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
      set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
    ENDIF()

    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
    set(CMAKE_CUDA_PATH /usr/local/cuda/)
    set(CUDA_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

    # without it we get "nvcc warning : The -std=c++20 flag is not supported with the configured host compiler. Flag will be ignored."
    # https://stackoverflow.com/q/77170793/7325599
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    message("CMAKE_CXX_COMPILER=CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")

    # If the following line gives you this error:  Could NOT find CUDA (missing: CUDA_NVCC_EXECUTABLE) (found suitable version "12.1", minimum required is "12")
    # That's because you ran this file twice. Make sure it runs at most once.
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

    set(CUDART_LIBRARY ${CUDA_LIBRARIES})
  ENDIF()
ENDIF()
