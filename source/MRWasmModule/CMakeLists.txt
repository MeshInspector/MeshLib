project(MRWasmModule CXX)

add_executable(${PROJECT_NAME} MRWasmBindings.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE "-fexceptions")

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    -Wl,--whole-archive
    MRMesh
    -Wl,--no-whole-archive
    ${CMAKE_DL_LIBS}
)

set(CMAKE_EXECUTABLE_SUFFIX ".mjs")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "meshlib")

target_link_options(${PROJECT_NAME} PRIVATE
  "--bind"
  "-fexceptions"
  "-sMODULARIZE=1"
  "-sEXPORT_NAME=createMeshLib"
  "-sEXPORT_ES6=1"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sEXIT_RUNTIME=0"
  "-sINVOKE_RUN=0"
)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION "${MR_BIN_DIR}"
)
install(
  FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/meshlib.wasm"
  DESTINATION "${MR_BIN_DIR}"
  OPTIONAL
)
