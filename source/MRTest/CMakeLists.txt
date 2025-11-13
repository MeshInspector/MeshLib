project(MRTest CXX)

file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

IF(EMSCRIPTEN)
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")
  set(EM_WHOLE_ARCHIVE "-Wl,--whole-archive")
  set(EM_NO_WHOLE_ARCHIVE "-Wl,--no-whole-archive")
ENDIF()

find_package(GTest REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRIOExtras
  ${EM_WHOLE_ARCHIVE}
  MRViewer
  ${EM_NO_WHOLE_ARCHIVE}
  GTest::gtest
  ${CMAKE_DL_LIBS}
)

IF(NOT EMSCRIPTEN)
  find_package(cpr REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE cpr::cpr)
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
