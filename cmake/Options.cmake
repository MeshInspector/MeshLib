IF(DEFINED ENV{MR_USE_CPP_23} AND "$ENV{MR_USE_CPP_23}" STREQUAL "ON")
  set(MR_CXX_STANDARD 23)
  message("enable c++23")
ELSE()
  set(MR_CXX_STANDARD 20)
ENDIF()
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(NOT MR_EMSCRIPTEN)
  IF(APPLE)
    message("building for Apple")
    execute_process(
      COMMAND brew --prefix
      RESULT_VARIABLE CMD_ERROR
      OUTPUT_VARIABLE HOMEBREW_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    IF(CMD_ERROR EQUAL 0 AND EXISTS "${HOMEBREW_PREFIX}")
      message("Homebrew found. Prefix: ${HOMEBREW_PREFIX}")
    ELSE()
      message("Homebrew not found!")
      message(FATAL_ERROR "${CMD_ERROR} ${HOMEBREW_PREFIX}")
    ENDIF()
    # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
    link_directories(${HOMEBREW_PREFIX}/lib)
    set(CPPFLAGS "-I${HOMEBREW_PREFIX}/opt/llvm/include -I${HOMEBREW_PREFIX}/include")
    set(LDFLAGS "-L${HOMEBREW_PREFIX}/opt/llvm/lib -Wl,-rpath,${HOMEBREW_PREFIX}/opt/llvm/lib")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -undefined dynamic_lookup -framework Cocoa -framework OpenGL -framework IOKit") # https://github.com/pybind/pybind11/issues/382

    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib")

    include_directories(${HOMEBREW_PREFIX}/include)
  ELSE()
    # such print format is selected to simplify load in Excel with fields (user time, system time, command) subdivision by commas
    IF(EXISTS /usr/bin/time)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "/usr/bin/time -f \",%U,%S,%C\" -a -o compile_timings.txt")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "/usr/bin/time -f \",%U,%S,%C\" -a -o link_timings.txt")
    ENDIF()
  ENDIF() # APPLE
ENDIF() # NOT MR_EMSCRIPTEN

message("CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}")
# for MacOS, GCC and Clang<15 builds: PCH not only does not give any speedup, but even vice versa
IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
  IF(NOT MR_EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  ENDIF()
  set(MR_PCH true CACHE BOOL "Enable precompiled headers.")
ElSE()
  set(MR_PCH false CACHE BOOL "Enable precompiled headers.")
ENDIF()
message("MR_PCH=${MR_PCH}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DDEBUG")
# turn on warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-sign-compare -Werror -fvisibility=hidden -pedantic-errors -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS -DIMGUI_ENABLE_FREETYPE")
IF(NOT MR_EMSCRIPTEN_SINGLETHREAD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD
IF(WIN32 AND MINGW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
ENDIF()
# make link to fail if there are unresolved symbols (GCC and Clang)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,defs")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs")
ENDIF()

IF(MR_EMSCRIPTEN)
  IF(MR_EMSCRIPTEN_SINGLETHREAD)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web")
  ELSE()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web,worker -pthread -s PTHREAD_POOL_SIZE_STRICT=0 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pthreads-mem-growth") # look https://github.com/emscripten-core/emscripten/issues/8287
  ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_RUNTIME_METHODS=[ccall] -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB -s LLD_REPORT_UNDEFINED=1 -s USE_WEBGL2=1 -s USE_GLFW=3 -s USE_ZLIB=1 -s FULL_ES3=1 -s USE_LIBPNG=1")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --preload-file ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets@/")
  IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASYNCIFY -s ASYNCIFY_IGNORE_INDIRECT -Wno-limited-postlink-optimizations")
    add_compile_definitions(MR_EMSCRIPTEN_ASYNCIFY)
  ENDIF() # NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY
  add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)
ENDIF() # MR_EMSCRIPTEN
