cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (meshconv CXX)

find_package(Boost COMPONENTS program_options REQUIRED )
if(Boost_PROGRAM_OPTIONS_FOUND)
    link_libraries( ${Boost_PROGRAM_OPTIONS_LIBRARY} )
endif()

add_executable(${PROJECT_NAME} meshconv.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
        MRMesh
        fmt
        spdlog
        Boost::boost
)
