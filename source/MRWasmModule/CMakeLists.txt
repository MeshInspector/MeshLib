project(MRWasmModule CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

# Compile the embind bindings once, then link them into both module variants below.
add_library(MRWasmModuleObjects OBJECT ${SOURCES} ${HEADERS})
target_link_libraries(MRWasmModuleObjects PRIVATE MRMesh)
if(TARGET MRVoxels)
  target_link_libraries(MRWasmModuleObjects PRIVATE MRVoxels)
endif()

set(CMAKE_EXECUTABLE_SUFFIX ".mjs")
if(MR_EMSCRIPTEN_SINGLETHREAD)
  set(MESHLIB_WASM_OUTPUT_NAME "meshlib")
else()
  set(MESHLIB_WASM_OUTPUT_NAME "meshlib-mt")
endif()

function(mr_add_wasm_module TARGET_NAME)
  cmake_parse_arguments(ARG "" "OUTPUT_NAME" "LINK_OPTIONS" ${ARGN})

  add_executable(${TARGET_NAME} $<TARGET_OBJECTS:MRWasmModuleObjects>)
  target_link_libraries(${TARGET_NAME}
    PRIVATE
      -Wl,--whole-archive
      MRMesh
      -Wl,--no-whole-archive
      ${CMAKE_DL_LIBS}
  )
  if(TARGET MRVoxels)
    target_link_libraries(${TARGET_NAME} PRIVATE -Wl,--whole-archive MRVoxels -Wl,--no-whole-archive)
  endif()

  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${ARG_OUTPUT_NAME}" LINKER_LANGUAGE CXX)
  target_link_options(${TARGET_NAME} PRIVATE
    "--bind"
    "SHELL:--emit-tsd ${ARG_OUTPUT_NAME}.d.mts"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createMeshLib"
    "-sEXPORT_ES6=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sEXIT_RUNTIME=0"
    "-sINVOKE_RUN=0"
    "--closure=1"
    ${ARG_LINK_OPTIONS}
  )

  install(
    TARGETS ${TARGET_NAME}
    DESTINATION "${MR_BIN_DIR}"
  )
  install(
    FILES
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ARG_OUTPUT_NAME}.wasm"
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ARG_OUTPUT_NAME}.d.mts"
    DESTINATION "${MR_BIN_DIR}"
    OPTIONAL
  )
endfunction()

mr_add_wasm_module(MRWasmModuleBrowser
  OUTPUT_NAME "${MESHLIB_WASM_OUTPUT_NAME}"
  LINK_OPTIONS "-sENVIRONMENT=web,worker"
)
mr_add_wasm_module(MRWasmModuleNode
  OUTPUT_NAME "${MESHLIB_WASM_OUTPUT_NAME}.node"
  LINK_OPTIONS
    "-sENVIRONMENT=node"
    "-sNODERAWFS=1"
)
