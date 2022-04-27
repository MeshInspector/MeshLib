cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (meshconv CXX)

IF (NOT MR_EMSCRIPTEN)
  find_package(Boost COMPONENTS REQUIRED )
ENDIF()

add_executable(${PROJECT_NAME} meshconv.cpp)

IF(NOT ${distro} STREQUAL "Fedora Linux")
 target_link_libraries(${PROJECT_NAME} PRIVATE
         mrealgorithmspy
         MREAlgorithms
         MRMesh
         spdlog
         mrmeshpy
         Boost::boost
 )
ELSE()
  IF (NOT MR_EMSCRIPTEN)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        mrealgorithmspy
        MREAlgorithms
        MRMesh
        fmt
        gtest
        spdlog
        mrmeshpy
        Boost::boost
    )
  ELSE()
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    target_link_libraries(${PROJECT_NAME} PRIVATE
        gtest
        -Wl,--whole-archive
        MREAlgorithms
        MRMesh
        -Wl,--no-whole-archive
    )    
  ENDIF()    
ENDIF()
