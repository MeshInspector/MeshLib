cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mrmeshpy CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
)
IF(WIN32 OR MR_EMSCRIPTEN)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
  )
ELSE()
  set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${MESHLIB_PROJECT_NAME}/meshlib/"
  )
ENDIF()

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    spdlog
    fmt
    pybind11::module
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PUBLIC
    MRMesh
    spdlog
    fmt
    tbb
    openvdb
    Python::Python
  )
ENDIF()

install(TARGETS ${PROJECT_NAME} DESTINATION "${MR_PY_LIB_DIR}")
install(TARGETS ${PROJECT_NAME} EXPORT MeshLib DESTINATION "${MR_MAIN_LIB_DIR}")

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
