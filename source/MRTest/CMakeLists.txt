project(MRTest CXX)

file(GLOB SOURCES "*.cpp")
list(REMOVE_ITEM SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/MRTestDiagnostics.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/MRTestDiagnosticsPython.cpp"
)

add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(MRTestDiagnostics MRTestDiagnostics.cpp)

find_package(GTest REQUIRED)
IF(NOT EMSCRIPTEN)
  find_package(cpr REQUIRED)
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      MRIOExtras
      MRViewer
      MRSymbolMesh
      cpr::cpr
      GTest::gtest
  )
  target_link_libraries(MRTestDiagnostics PRIVATE MRViewer)
ELSE()
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      -Wl,--whole-archive
      MRViewer
      -Wl,--no-whole-archive
      MRSymbolMesh
      GTest::gtest
  )
  target_link_libraries(MRTestDiagnostics
    PRIVATE
      -Wl,--whole-archive
      MRViewer
      -Wl,--no-whole-archive
  )
ENDIF()

if(WIN32)
  include(MimallocRedirect)
  mr_win32_enable_mimalloc_redirect(${PROJECT_NAME})
endif()

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
  add_executable(MRTestDiagnosticsPython MRTestDiagnosticsPython.cpp)
  target_link_libraries(MRTestDiagnosticsPython PRIVATE
    MRMesh
    MRPython
    MREmbeddedPython
    ${CMAKE_DL_LIBS}
  )
  IF(NOT WIN32)
    target_link_libraries(MRTestDiagnosticsPython PRIVATE Python::Python)
  ENDIF()
  IF(MR_PCH)
    target_precompile_headers(MRTestDiagnosticsPython REUSE_FROM MRPch)
  ENDIF()
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
  target_precompile_headers(MRTestDiagnostics REUSE_FROM MRPch)
ENDIF()
