cmake_minimum_required(VERSION 3.16)
project(MREmbreeBench CXX)

# Embree is built and installed separately with its own default flags (pass embree_DIR),
# against the same TBB that MeshLib uses
find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)
find_package(embree 4 REQUIRED CONFIG)

add_executable(${PROJECT_NAME} MREmbreeBench.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE MRMesh embree TBB::tbb)
