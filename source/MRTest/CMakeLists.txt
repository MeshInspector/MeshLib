cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (MRTest CXX)

add_executable(${PROJECT_NAME} MRTestApp.cpp)

IF(NOT ${distro} STREQUAL "Fedora Linux")
 target_link_libraries(${PROJECT_NAME} PRIVATE
         MRMesh
         spdlog
         mrmeshpy
 )
ELSE()
  IF (NOT MR_EMSCRIPTEN)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        MRMesh
        fmt
        gtest
        spdlog
        mrmeshpy
    )
  ELSE()
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    target_link_libraries(${PROJECT_NAME} PRIVATE
        gtest
        -Wl,--whole-archive
        MRMesh
        -Wl,--no-whole-archive
    )    
  ENDIF()    
ENDIF()

add_test(NAME    ${PROJECT_NAME}
         COMMAND ${PROJECT_NAME})
