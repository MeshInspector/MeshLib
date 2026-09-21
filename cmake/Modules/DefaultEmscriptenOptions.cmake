# this file must be included BEFORE the `project' command: MEMORY64 has to be in effect while CMake probes the compiler

if(MR_EMSCRIPTEN)
  if(MR_EMSCRIPTEN_WASM64)
    # emcc accepts -m64 from 5.0.7 on and deprecates -s MEMORY64 in 6.0, where -Wdeprecated
    # -Werror turns the deprecation into an error that fails every compiler probe. Older SDKs
    # pass -m64 through to clang and fail just as hard, so the spelling has to follow the SDK.
    # This runs before project(), so the toolchain file has not set EMSCRIPTEN_VERSION yet;
    # derive it the same way it does. It only sets the variable if we leave it empty.
    if(NOT EMSCRIPTEN_VERSION)
      find_program(MESHLIB_EMCC NAMES emcc HINTS "$ENV{EMSDK}/upstream/emscripten" REQUIRED)
      execute_process(COMMAND "${MESHLIB_EMCC}" -v ERROR_VARIABLE MESHLIB_EMCC_OUTPUT OUTPUT_QUIET)
      string(REGEX MATCH "emcc [(].*[)] ([0-9.]+)" MESHLIB_EMCC_UNUSED "${MESHLIB_EMCC_OUTPUT}")
      if(NOT CMAKE_MATCH_1)
        message(FATAL_ERROR "Cannot parse the Emscripten version from \"${MESHLIB_EMCC} -v\"")
      endif()
      set(EMSCRIPTEN_VERSION "${CMAKE_MATCH_1}")
    endif()
    if(EMSCRIPTEN_VERSION VERSION_LESS "5.0.7")
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
