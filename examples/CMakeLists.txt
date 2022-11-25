cmake_minimum_required(VERSION 3.22)
project(test)

set(CMAKE_CXX_STANDARD 20)

add_executable(test main.cpp)

find_package(meshlib REQUIRED)
target_link_libraries(test PUBLIC MRMesh)
