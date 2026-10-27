# The WebAssembly module of the generated JS bindings: the C bindings and MeshLib linked into one Emscripten module,
# exporting every C function the generated JS calls (listed by the generator in `js/exported_functions.txt`).
# The generated JS (`js/`, from `make -f scripts/mrbind/generate.mk TARGET=js`) imports the module as `./MeshLibC2.mjs`,
# so the module is written next to it.

project(MeshLibJS CXX)

set(MESHLIB_JS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/js")
if(NOT EXISTS "${MESHLIB_JS_DIR}/emcc_link_flags.rsp")
  message(FATAL_ERROR "The JS bindings aren't generated: `${MESHLIB_JS_DIR}/emcc_link_flags.rsp` is missing. Run `make -f scripts/mrbind/generate.mk TARGET=js` first.")
endif()

# An empty source: everything comes from the libraries.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/MeshLibJS_empty.cpp" "")
add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_BINARY_DIR}/MeshLibJS_empty.cpp")

# The C bindings must be linked whole, the exported functions are referenced by nothing else.
# `MeshLibC2Cuda` holds the Cuda stubs (`MR_Cuda_isCudaAvailable()`), which the generated JS calls too.
target_link_libraries(${PROJECT_NAME} PRIVATE
  -Wl,--whole-archive
  MeshLibC2
  MeshLibC2Cuda
  -Wl,--no-whole-archive
  MRMesh
  MRIOExtras
  MRSymbolMesh
  MRVoxels
  ${CMAKE_DL_LIBS}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  OUTPUT_NAME "MeshLibC2"
  SUFFIX ".mjs"
  LINKER_LANGUAGE CXX
  RUNTIME_OUTPUT_DIRECTORY "${MESHLIB_JS_DIR}"
)
target_link_options(${PROJECT_NAME} PRIVATE
  "SHELL:@${MESHLIB_JS_DIR}/emcc_link_flags.rsp"
  "SHELL:-sEXPORTED_FUNCTIONS=@${MESHLIB_JS_DIR}/exported_functions.txt"
  "-sMODULARIZE=1"
  "-sEXPORT_ES6=1"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sEXIT_RUNTIME=0"
  "-sINVOKE_RUN=0"
  "-sENVIRONMENT=web,worker,node"
  "-sNODERAWFS=0"
)
