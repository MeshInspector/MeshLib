cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(meshconv CXX)

find_package(Boost COMPONENTS program_options REQUIRED)
IF(Boost_PROGRAM_OPTIONS_FOUND)
  link_libraries(${Boost_PROGRAM_OPTIONS_LIBRARY})
ENDIF()

add_executable(${PROJECT_NAME} meshconv.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  fmt
  spdlog
  Boost::boost
  ${MR_PYTHON_LIB}
)

install(TARGETS ${PROJECT_NAME} DESTINATION "${MR_BIN_DIR}")
install(TARGETS ${PROJECT_NAME} EXPORT MeshLib DESTINATION "${MR_MAIN_LIB_DIR}")

IF(NOT APPLE)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
