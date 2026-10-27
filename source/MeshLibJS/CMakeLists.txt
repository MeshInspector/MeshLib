# The WebAssembly modules of the generated JS bindings: the C bindings and MeshLib linked into one Emscripten module,
# exporting every C function the generated JS calls (listed by the generator in `exported_functions.txt`).
# The generated JS (`js/`, or `js64/` for `-sMEMORY64`, from `make -f scripts/mrbind/generate.mk TARGET=js [JS_WASM64=1]`)
# imports the modules as `./MeshLibC2.mjs` (browsers and workers) and `./MeshLibC2.node.mjs` (Node, with the real file system),
# so it's copied into the build tree (`<build>/MeshLibJS/js` or `js64`) and the modules are written next to the copy.

project(MeshLibJS CXX)

# The libraries this links (some are added to the build after this directory, so the options are checked, not the targets).
foreach(opt MESHLIB_BUILD_EXTRA_IO_FORMATS MESHLIB_BUILD_SYMBOLMESH MESHLIB_BUILD_VOXELS)
  if(NOT ${opt})
    message(FATAL_ERROR "The JS bindings need ${opt}=ON.")
  endif()
endforeach()

if(MR_EMSCRIPTEN_WASM64)
  set(MESHLIB_JS_SUBDIR "js64")
else()
  set(MESHLIB_JS_SUBDIR "js")
endif()
set(MESHLIB_JS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${MESHLIB_JS_SUBDIR}")
set(MESHLIB_JS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${MESHLIB_JS_SUBDIR}")
if(NOT EXISTS "${MESHLIB_JS_DIR}/emcc_link_flags.rsp")
  message(FATAL_ERROR "The JS bindings aren't generated: `${MESHLIB_JS_DIR}/emcc_link_flags.rsp` is missing. Run `make -f scripts/mrbind/generate.mk TARGET=js` (with `JS_WASM64=1` for MR_EMSCRIPTEN_WASM64) first.")
endif()

add_custom_target(MeshLibJSSources
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${MESHLIB_JS_DIR}" "${MESHLIB_JS_OUTPUT_DIR}"
  COMMENT "Copying the generated JS bindings to ${MESHLIB_JS_OUTPUT_DIR}"
)

# An empty source: everything comes from the libraries.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/MeshLibJS_empty.cpp" "")

function(mr_add_js_module TARGET_NAME)
  cmake_parse_arguments(ARG "" "OUTPUT_NAME" "LINK_OPTIONS" ${ARGN})

  add_executable(${TARGET_NAME} "${CMAKE_CURRENT_BINARY_DIR}/MeshLibJS_empty.cpp")

  # The C bindings must be linked whole, the exported functions are referenced by nothing else.
  # `MeshLibC2Cuda` holds the Cuda stubs (`MR_Cuda_isCudaAvailable()`), which the generated JS calls too.
  target_link_libraries(${TARGET_NAME} PRIVATE
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

  add_dependencies(${TARGET_NAME} MeshLibJSSources)
  set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${ARG_OUTPUT_NAME}"
    SUFFIX ".mjs"
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${MESHLIB_JS_OUTPUT_DIR}"
  )
  # The generated link flags export Emscripten's exception helpers (our C bindings don't catch C++ exceptions, the JS converts
  # the escaping ones), which need the exception-enabled system libraries: `-fexceptions` at link time selects them.
  target_link_options(${TARGET_NAME} PRIVATE
    "-fexceptions"
    "SHELL:@${MESHLIB_JS_DIR}/emcc_link_flags.rsp"
    "SHELL:-sEXPORTED_FUNCTIONS=@${MESHLIB_JS_DIR}/exported_functions.txt"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sEXIT_RUNTIME=0"
    "-sINVOKE_RUN=0"
    ${ARG_LINK_OPTIONS}
  )
endfunction()

mr_add_js_module(MeshLibJS
  OUTPUT_NAME "MeshLibC2"
  LINK_OPTIONS "-sENVIRONMENT=web,worker"
)
mr_add_js_module(MeshLibJSNode
  OUTPUT_NAME "MeshLibC2.node"
  LINK_OPTIONS "-sENVIRONMENT=node" "-sNODERAWFS=1"
)
