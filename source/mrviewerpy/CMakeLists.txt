cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mrviewerpy CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
)

IF(WIN32)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    SUFFIX ".pyd"
  )

  # We build directly in `bin`, then copy to `bin/meshlib`, to prevent Vcpkg from polluting `bin/meshlib` with the dependency DLLs.
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
  )
ENDIF()

IF(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRViewer
    imgui
    MRMesh
    MRPython
    fmt
    glfw
    spdlog
    pybind11::module
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRViewer
    imgui
    MRMesh
    MRPython
    fmt::fmt
    glfw
    spdlog::spdlog
    TBB::tbb
    ${CMAKE_DL_LIBS}
  )
  IF(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE pybind11nonlimitedapi_stubs)
  ELSE()
    find_package(pybind11nonlimitedapi REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE pybind11nonlimitedapi::pybind11nonlimitedapi_stubs)
  ENDIF()
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRViewer
    imgui
    MRMesh
    MRPython
    JsonCpp::JsonCpp
    fmt
    glfw
    spdlog
    tbb
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ENDIF()

# Don't emit undefined references. We need this because we don't want to link to a specific Python version.
# The flag we're removing here is something we only set on Linux, in `CompilerOptions.cmake`.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

IF(MESHLIB_BUILD_VOXELS)
  IF(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRVoxels
      OpenVDB
    )
  ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
    find_package(OpenVDB REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRVoxels
      OpenVDB::openvdb
    )
  ELSE()
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRVoxels
      openvdb
    )
  ENDIF()
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

install(TARGETS ${PROJECT_NAME} DESTINATION "${MR_PY_LIB_DIR}")

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
