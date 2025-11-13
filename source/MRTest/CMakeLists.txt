project(MRTest CXX)

file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

IF(EMSCRIPTEN)
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")

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
ELSE()
  find_package(GTest REQUIRED)
  find_package(cpr REQUIRED)

  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    MRIOExtras
    MRViewer
    cpr::cpr
    GTest::gtest
    ${CMAKE_DL_LIBS}
  )
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)

IF(MESHLIB_BUILD_VOXELS)
  target_link_libraries(${PROJECT_NAME} PRIVATE MRVoxels)
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

IF(MESHLIB_PYTHON_SUPPORT)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MREmbeddedPython
    MRPython
  )
  IF(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Python::Python)
  ENDIF()
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_PYTHON)
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
