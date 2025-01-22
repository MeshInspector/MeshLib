IF(MR_EMSCRIPTEN)
  # reference: https://github.com/emscripten-core/emscripten/blob/main/src/settings.js
  string(JOIN " " CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS}"
    "-s EXPORTED_RUNTIME_METHODS=[ccall]"
    "-s ALLOW_MEMORY_GROWTH=1"
    "-s LLD_REPORT_UNDEFINED=1"
    "-s USE_WEBGL2=1"
    "-s USE_GLFW=3"
    "-s USE_ZLIB=1"
    "-s FULL_ES3=1"
    "-s USE_LIBPNG=1"
  )

  IF(MR_EMSCRIPTEN_SINGLETHREAD)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web")
  ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web,worker -pthread -s PTHREAD_POOL_SIZE_STRICT=0 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
    
    # uncomment to enable source map for debugging in browsers (slow)
    #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gsource-map")

    IF(MR_EMSCRIPTEN_WASM64)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s MEMORY64=1") # required for correct platform detection
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s MEMORY64=1")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MEMORY64=1 -s MAXIMUM_MEMORY=16GB") # wasm-ld: maximum memory [...] cannot be greater than 17179869184
    ELSE()
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MAXIMUM_MEMORY=4GB")
    ENDIF()
  ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD

  IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASYNCIFY -Wno-limited-postlink-optimizations")
    add_compile_definitions(MR_EMSCRIPTEN_ASYNCIFY)
  ENDIF() # NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY

  add_compile_definitions(SPDLOG_FMT_EXTERNAL)
  add_compile_definitions(SPDLOG_WCHAR_FILENAMES) # hack to make it work with new version of fmt
  
  # FIXME: comment required
  add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)
ENDIF() # MR_EMSCRIPTEN
