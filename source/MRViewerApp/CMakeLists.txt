project(MeshViewer CXX)

add_executable(${PROJECT_NAME} WIN32 MRViewerApp.cpp)

IF(NOT EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      MRMesh
      MRViewer
  )
ELSE()
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      -Wl,--whole-archive
      MRMesh
      MRIOExtras
      MRViewer
      MRCommonPlugins
      -Wl,--no-whole-archive
  )
ENDIF()

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION "${MR_BIN_DIR}"
)

file(GLOB PNGS "*.png")
file(
  COPY ${PNGS}
  DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
install(
  FILES ${PNGS}
  DESTINATION "${MR_RESOURCES_DIR}"
)

IF(EMSCRIPTEN)
  file(GLOB LOCAL_WASM_FILES "../../wasm/*.*")
  file(
    COPY ${LOCAL_WASM_FILES}
    DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
  install(
    FILES ${LOCAL_WASM_FILES}
    DESTINATION "${MR_RESOURCES_DIR}/wasm"
  )
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
