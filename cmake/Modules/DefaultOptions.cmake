# this file must be included BEFORE the `project' command because the compiler flags are crucial for the platform detection

IF(DEFINED ENV{MR_USE_CPP_23} AND "$ENV{MR_USE_CPP_23}" STREQUAL "ON")
  message("MR_USE_CPP_23 variable is deprecated; consider setting MR_CXX_STANDARD to 23")
  set(MR_CXX_STANDARD 23 CACHE STRING "Version of the C++ standard used to compile the project")
ELSE()
  set(MR_CXX_STANDARD 20 CACHE STRING "Version of the C++ standard used to compile the project")
ENDIF()
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(MR_EMSCRIPTEN AND MR_EMSCRIPTEN_WASM64)
  # required to be set before `project()' command for correct platform detection
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s MEMORY64=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s MEMORY64=1")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MEMORY64=1")
ENDIF()

add_compile_definitions(MR_USE_CMAKE_CONFIGURE_FILE)
