# Shared stage setup that must run BEFORE project(), because it decides how the compiler is
# probed: the language standard, the Emscripten memory model and the vcpkg toolchain.
#
# CMake requires a literal, direct project() call in the top-level CMakeLists.txt -- calling it
# from an included file makes CMake inject an implicit project() at line 1, which probes the
# compiler before these flags are set. That is why this file and common_post_project.cmake are
# separate rather than one common.cmake.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

set(MESHLIB_THIRDPARTY_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MESHLIB_THIRDPARTY_INCLUDE_DIR "include")

list(APPEND CMAKE_MODULE_PATH "${MESHLIB_THIRDPARTY_DIR}/../cmake/Modules")
include(ConfigureVcpkg)

IF(MR_EMSCRIPTEN_WASM64)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s MEMORY64=1")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s MEMORY64=1")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MEMORY64=1")
ENDIF()
