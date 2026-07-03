project(MRWasmModule CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    -Wl,--whole-archive
    MRMesh
    -Wl,--no-whole-archive
    ${CMAKE_DL_LIBS}
)

if(TARGET MRVoxels)
  target_link_libraries(${PROJECT_NAME} PRIVATE -Wl,--whole-archive MRVoxels -Wl,--no-whole-archive)
endif()

set(CMAKE_EXECUTABLE_SUFFIX ".mjs")
if(MR_EMSCRIPTEN_SINGLETHREAD)
  set(MESHLIB_WASM_OUTPUT_NAME "meshlib")
else()
  set(MESHLIB_WASM_OUTPUT_NAME "meshlib-mt")
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${MESHLIB_WASM_OUTPUT_NAME}")

target_link_options(${PROJECT_NAME} PRIVATE
  "--bind"
  "-sMODULARIZE=1"
  "-sEXPORT_NAME=createMeshLib"
  "-sEXPORT_ES6=1"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sEXIT_RUNTIME=0"
  "-sINVOKE_RUN=0"
  # keep the module Asyncify-free even when co-built with MRViewer
  "-sASYNCIFY=0"
)

set(MESHLIB_WASM_MODULE_TARGET "node" CACHE STRING "Target runtime for the WASM module: 'node' or 'web'")
set_property(CACHE MESHLIB_WASM_MODULE_TARGET PROPERTY STRINGS "node" "web")
if(MESHLIB_WASM_MODULE_TARGET STREQUAL "node")
  target_link_options(${PROJECT_NAME} PRIVATE "-sNODERAWFS=1")
endif()

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION "${MR_BIN_DIR}"
)
install(
  FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${MESHLIB_WASM_OUTPUT_NAME}.wasm"
  DESTINATION "${MR_BIN_DIR}"
  OPTIONAL
)
