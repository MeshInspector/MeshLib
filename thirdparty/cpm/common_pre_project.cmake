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

get_filename_component(MESHLIB_THIRDPARTY_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
set(MESHLIB_THIRDPARTY_INCLUDE_DIR "include")

list(APPEND CMAKE_MODULE_PATH "${MESHLIB_THIRDPARTY_DIR}/../cmake/Modules")
include(ConfigureVcpkg)
if(MR_EMSCRIPTEN)
  include(DefaultEmscriptenOptions)
endif()
