# this file must be included BEFORE the `project' command: MEMORY64/m64 has to be in effect while CMake probes the compiler

# neither EMSCRIPTEN nor EMSCRIPTEN_VERSION are set until the Emscripten toolchain is loaded
if(MR_EMSCRIPTEN)
  if(NOT EMSCRIPTEN_VERSION)
    find_program(EMCC NAMES emcc HINTS "$ENV{EMSDK}/upstream/emscripten" REQUIRED)
    execute_process(COMMAND "${EMCC}" -v ERROR_VARIABLE EMCC_OUTPUT OUTPUT_QUIET)
    string(REGEX MATCH "emcc \\(.*\\) ([0-9\\.]+)" EMCC_OUTPUT "${EMCC_OUTPUT}")
    if(NOT CMAKE_MATCH_1)
      message(FATAL_ERROR "Cannot parse the Emscripten version")
    endif()
    unset(EMCC)
    unset(EMCC_OUTPUT)
    set(EMSCRIPTEN_VERSION "${CMAKE_MATCH_1}")
  endif()

  if(MR_EMSCRIPTEN_WASM64)
    if(EMSCRIPTEN_VERSION VERSION_LESS "6.0.0")
      set(MESHLIB_EMSCRIPTEN_WASM64_FLAG "-s MEMORY64=1")
    else()
      set(MESHLIB_EMSCRIPTEN_WASM64_FLAG "-m64")
    endif()
    string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}
      "${MESHLIB_EMSCRIPTEN_WASM64_FLAG}"
    )
    string(JOIN " " MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}
      "${MESHLIB_EMSCRIPTEN_WASM64_FLAG}"
    )
  endif()

  if(NOT MR_EMSCRIPTEN_SINGLETHREAD)
    string(JOIN " " MESHLIB_EMSCRIPTEN_CXX_FLAGS ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}
      "-pthread"
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

  string(APPEND CMAKE_C_FLAGS " ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")
  string(APPEND CMAKE_CXX_FLAGS " ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}")
endif()
