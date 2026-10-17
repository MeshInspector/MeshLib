cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MRBenchCollide CXX)

find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)
add_executable(${PROJECT_NAME} MRBenchCollide.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE MRMesh TBB::tbb)
