cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (mrviewerpy CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

if(APPLE)
	set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib")

find_package(pybind11)

target_link_libraries(${PROJECT_NAME} PRIVATE
	MRViewer
	MRMesh
	fmt
	pybind11::module
)

