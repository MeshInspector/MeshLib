# reference: https://github.com/emscripten-core/emscripten/blob/main/src/settings.js
string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS
  "--use-port=boost_headers"
  "--use-port=freetype" # TODO: make optional
  "--use-port=libpng" # TODO: make optional
  "--use-port=zlib" # TODO: make optional
)
string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS
  "-s EXPORTED_RUNTIME_METHODS=[ccall]"
  "-s ALLOW_MEMORY_GROWTH=1"
  "-s LLD_REPORT_UNDEFINED=1"
  "-s USE_WEBGL2=1"
  "-s USE_GLFW=3"
  "-s FULL_ES3=1"
)

IF(MR_EMSCRIPTEN_SINGLETHREAD)
  set(MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS "${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS} -s ENVIRONMENT=web,node")
ELSE()
  string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}
    "-pthread"
    # look https://github.com/emscripten-core/emscripten/issues/8287
    "-Wno-pthreads-mem-growth"
  )

  # uncomment to enable source map for debugging in browsers (slow)
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gsource-map")

  IF(MR_EMSCRIPTEN_WASM64)
    set(MAXIMUM_MEMORY 16GB) # wasm-ld: maximum memory [...] cannot be greater than 17179869184
  ELSE()
    set(MAXIMUM_MEMORY 4GB)
  ENDIF()
  string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}
    "-s ENVIRONMENT=web,worker,node"
    #"-pthread"
    "-s PTHREAD_POOL_SIZE_STRICT=0"
    "-s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency"
    "-s MAXIMUM_MEMORY=${MAXIMUM_MEMORY}"
  )
ENDIF()

IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
  string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}
    "-s ASYNCIFY"
    # FIXME: comment required
    "-Wno-limited-postlink-optimizations"
  )

  add_compile_definitions(MR_EMSCRIPTEN_ASYNCIFY)
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}")
