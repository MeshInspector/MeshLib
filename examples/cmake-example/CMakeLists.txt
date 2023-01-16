cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cmake-example)

add_executable(${PROJECT_NAME} main.cpp)

find_package(MeshLib CONFIG REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})

find_package(Boost COMPONENTS REQUIRED )

target_link_libraries(${PROJECT_NAME} MeshLib::MRMesh Boost::boost)