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
    MRPython
    MREmbeddedPython
    MRViewer
    jsoncpp
    fmt
    gtest
    spdlog
    Python::Python
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ELSE()
  IF(MR_EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    target_link_libraries(${PROJECT_NAME} PRIVATE
      gtest
      zip
      jsoncpp
      -Wl,--whole-archive
      MRMesh
      MRIOExtras
      MRViewer
      -Wl,--no-whole-archive
    )
  ELSEIF(WIN32)
    find_package(GTest REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRMesh
      MRIOExtras
      MRPython
      MREmbeddedPython
      MRViewer
      JsonCpp::JsonCpp
      fmt::fmt
      GTest::gtest
      spdlog::spdlog
      pybind11nonlimitedapi_stubs
      ${CMAKE_DL_LIBS}
    )
  ELSE()
    find_package(GTest REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRMesh
      MRIOExtras
      MRPython
      MREmbeddedPython
      MRViewer
      JsonCpp::JsonCpp
      fmt::fmt
      GTest::gtest
      spdlog::spdlog
      Python::Python
      pybind11nonlimitedapi_stubs
      ${CMAKE_DL_LIBS}
    )
  ENDIF()
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
