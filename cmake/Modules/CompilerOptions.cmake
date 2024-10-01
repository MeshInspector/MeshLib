set(MR_CXX_STANDARD 20 CACHE STRING "Version of the C++ standard used to compile the project")
IF(DEFINED ENV{MR_USE_CPP_23} AND "$ENV{MR_USE_CPP_23}" STREQUAL "ON")
  message("MR_USE_CPP_23 variable is deprecated; consider setting MR_CXX_STANDARD to 23")
  set(MR_CXX_STANDARD 23)
ENDIF()
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MR_PCH_DEFAULT OFF)
# for macOS, GCC, and Clang<15 builds: PCH not only does not give any speedup, but even vice versa
IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
  set(MR_PCH_DEFAULT ON)
#ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#  set(MR_PCH_DEFAULT ON)
ENDIF()
set(MR_PCH ${MR_PCH_DEFAULT} CACHE BOOL "Enable precompiled headers")
IF(MR_PCH AND NOT MR_EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
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

# TODO: __aarch64__ ?
IF(NOT APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  message("CMAKE_SYSTEM_PROCESSOR is ${CMAKE_SYSTEM_PROCESSOR}")
  add_compile_definitions(__ARM_CPU__)
  message("ARM cpu detected")
ENDIF()

IF(MR_EMSCRIPTEN)
  # reference: https://github.com/emscripten-core/emscripten/blob/main/src/settings.js
  string(JOIN " " CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS}"
    "-s EXPORTED_RUNTIME_METHODS=[ccall]"
    "-s ALLOW_MEMORY_GROWTH=1"
    "-s MAXIMUM_MEMORY=4GB"
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
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web,worker -pthread -s PTHREAD_POOL_SIZE_STRICT=0 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pthreads-mem-growth") # look https://github.com/emscripten-core/emscripten/issues/8287
  ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD

  IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASYNCIFY -Wno-limited-postlink-optimizations")
    add_compile_definitions(MR_EMSCRIPTEN_ASYNCIFY)
  ENDIF() # NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY

  # FIXME: comment required
  add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)
ENDIF() # MR_EMSCRIPTEN
