IF(NOT CUDAToolkit_FOUND AND NOT CUDA_FOUND)
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
ENDIF()
