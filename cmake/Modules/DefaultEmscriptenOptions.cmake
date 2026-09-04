# this file must be included BEFORE the `project' command: MEMORY64 has to be in effect while CMake probes the compiler

if(MR_EMSCRIPTEN)
  if(MR_EMSCRIPTEN_WASM64)
    string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}
      "-s MEMORY64=1"
    )
    string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}
      "-s MEMORY64=1"
    )
  endif()

  if(NOT MR_EMSCRIPTEN_SINGLETHREAD)
    string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}
      "-pthread"
      # look https://github.com/emscripten-core/emscripten/issues/8287
      "-Wno-pthreads-mem-growth"
    )
  endif()

  option(MR_EMSCRIPTEN_WASM2023 "Enable Unity's WebAssembly 2023 target (a set of general-purpose optimizations, including SIMD)" ON)
  if(MR_EMSCRIPTEN_WASM2023)
    # Those flags come from here: https://docs.unity3d.com/6000.7/Documentation/Manual/webgl-native-plugins-with-emscripten.html
    # Skipping `-fwasm-exceptions` because we don't use exceptions.
    # Skipping `-sSUPPORT_LONGJMP=wasm` because that conflicts with our `-s NO_DISABLE_EXCEPTION_CATCHING=1`, and also prevents CMake from finding FreeType during configuration.
    #   In theory, this flag is supposed to be implemented in terms of `-fwasm-exceptions`, so I'm not sure how it works without that one, but it seems to work (other than the issues above).
    #   Either way, we don't use `longjmp()`, so it doesn't seem terribly useful.
    string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}
      "-msimd128"
      "-mbulk-memory"
      "-mnontrapping-fptoint"
      "-msse4.2"
    )
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}")
endif()
