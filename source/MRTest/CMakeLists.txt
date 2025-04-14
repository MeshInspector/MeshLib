cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRTest CXX)

file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

IF(MR_EMSCRIPTEN)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")
ENDIF()

IF(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    MRIOExtras
    MRViewer
    jsoncpp
    fmt
    gtest
    spdlog
    OpenVDB
    tbb
    ${CMAKE_DL_LIBS}
  )
ELSE()
  IF(MR_EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    target_link_libraries(${PROJECT_NAME} PRIVATE
      gtest
      zip
      jsoncpp
      openvdb
      tbb
      -Wl,--whole-archive
      MRMesh
      MRIOExtras
      MRViewer
      -Wl,--no-whole-archive
    )
  ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
    find_package(GTest REQUIRED)
    find_package(OpenVDB REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRMesh
      MRIOExtras
      MRViewer
      JsonCpp::JsonCpp
      fmt::fmt
      GTest::gtest
      spdlog::spdlog
      OpenVDB::openvdb
      TBB::tbb
      ${CMAKE_DL_LIBS}
    )
  ELSE()
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRMesh
      MRIOExtras
      MRViewer
      JsonCpp::JsonCpp
      fmt
      gtest
      spdlog
      openvdb
      tbb
      ${CMAKE_DL_LIBS}
    )
  ENDIF()
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)

IF(MESHLIB_BUILD_VOXELS)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRVoxels
  )
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

IF(MESHLIB_PYTHON_SUPPORT)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRPython
    MREmbeddedPython
  )
  IF(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE pybind11nonlimitedapi_stubs)
  ELSE()
    find_package(pybind11nonlimitedapi REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Python::Python pybind11nonlimitedapi::pybind11nonlimitedapi_stubs)
  ENDIF()
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_PYTHON)
ENDIF()

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
