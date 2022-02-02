cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (MRTest CXX)

set(CMAKE_EXECUTABLE_SUFFIX ".html")

add_executable(${PROJECT_NAME} MRTestApp.cpp)

#IF(NOT ${distro} STREQUAL "Fedora Linux")
# target_link_libraries(${PROJECT_NAME} PRIVATE
#	 	 mrealgorithmspy
#	 	 MREAlgorithms
#         MRMesh
#         spdlog
#         mrmeshpy
# )
#ELSE()
 target_link_libraries(${PROJECT_NAME} PRIVATE -Wl,--whole-archive
 	 	 #mrealgorithmspy
         gtest
	 MREAlgorithms
         MRMesh -Wl,--no-whole-archive
         #fmt
         #spdlog
         #mrmeshpy
 )
#ENDIF()

add_test(NAME    ${PROJECT_NAME}
         COMMAND ${PROJECT_NAME})
