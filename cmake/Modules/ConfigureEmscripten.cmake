# reference: https://github.com/emscripten-core/emscripten/blob/main/src/settings.js
string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS
  # I would use the new-style flag spelling here, but it doesn't work on the old EMSDK 3.1.38 that we
  #   have to support for Unity compatibility.
  #"--use-port=boost_headers"
  #"--use-port=freetype" # TODO: make optional
  #"--use-port=libpng" # TODO: make optional
  #"--use-port=zlib" # TODO: make optional
  "-sUSE_BOOST_HEADERS"
  "-sUSE_FREETYPE" # TODO: make optional
  "-sUSE_LIBPNG" # TODO: make optional
  "-sUSE_ZLIB" # TODO: make optional
)

option(MR_EMSCRIPTEN_WASM2023 "Enable Unity's WebAssembly 2023 target (a set of general-purpose optimizations, including SIMD)" ON)
IF(MR_EMSCRIPTEN_WASM2023)
  # Those flags come from here: https://docs.unity3d.com/6000.7/Documentation/Manual/webgl-native-plugins-with-emscripten.html
  # Skipping `-fwasm-exceptions` because we don't use exceptions.
  # Skipping `-sSUPPORT_LONGJMP=wasm` because that conflicts with our `-s NO_DISABLE_EXCEPTION_CATCHING=1`. That conflict only happens in third-party libraries, not here, but still disabling it here for consistency.
  #   In theory, this flag is supposed to be implemented in terms of `-fwasm-exceptions`, so I'm not sure how it works without that one, but it seems to work if enabled.
  #   Either way, we don't juse `longjmp()`, so it doesn't seem terribly useful.
  string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}
    "-msimd128 -mbulk-memory -mnontrapping-fptoint -msse4.2 -sSUPPORT_LONGJMP=wasm"
  )
ENDIF()
string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS
  "-s ALLOW_MEMORY_GROWTH=1"
  "-s LLD_REPORT_UNDEFINED=1"
  "-s STACK_SIZE=1048576" # required for GDCM
)

option(MR_EMSCRIPTEN_MIMALLOC "Use mimalloc allocator (-s MALLOC=mimalloc) for Emscripten builds" ON)
IF(MR_EMSCRIPTEN_MIMALLOC)
  set(MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS "${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS} -s MALLOC=mimalloc")
ENDIF()

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

IF(MESHLIB_BUILD_MRVIEWER)
  string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}
    "-s EXPORTED_RUNTIME_METHODS=[ccall]"
    "-s USE_WEBGL2=1"
    "-s USE_GLFW=3"
    "-s FULL_ES3=1"
  )

  IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
    string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}
      "-s ASYNCIFY"
      # FIXME: comment required
      "-Wno-limited-postlink-optimizations"
    )
  ENDIF()
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}")
