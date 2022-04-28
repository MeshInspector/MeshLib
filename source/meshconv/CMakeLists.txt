cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (meshconv CXX)

find_package(Boost COMPONENTS program_options REQUIRED )

add_executable(${PROJECT_NAME} meshconv.cpp)

IF(${distro} STREQUAL "Fedora Linux")
  target_link_libraries(${PROJECT_NAME} PRIVATE
          MREAlgorithms
          MRMesh
          spdlog
          Boost::boost
          )
ELSE() # Ubuntu and macOS
  target_link_libraries(${PROJECT_NAME} PRIVATE
          MREAlgorithms
          MRMesh
          fmt
          spdlog
          Boost::boost
          )
ENDIF()
