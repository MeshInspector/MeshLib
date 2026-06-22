# Precompiled-header configuration.
# Included from the top-level CMakeLists right after CompilerOptions, so it must come AFTER the
# `project' command (it relies on the detected compiler information).

set(MR_PCH_DEFAULT ON)
# for Clang<15 builds: PCH not only does not give any speedup, but even vice versa
IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
  set(MR_PCH_DEFAULT OFF)
ENDIF()
set(MR_PCH ${MR_PCH_DEFAULT} CACHE BOOL "Enable precompiled headers")
IF(MR_PCH AND NOT MR_EMSCRIPTEN AND NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF()
message("MR_PCH=${MR_PCH}")

option(MR_PCH_USE_EXTRA_HEADERS "Add frequently used MeshLib headers to the precompiled header" OFF)

IF(MR_PCH AND MR_PCH_USE_EXTRA_HEADERS)
  add_compile_definitions(MR_PCH_USE_EXTRA_HEADERS)
ENDIF()

# Linux: enable LFS globally
# many Linux libraries' header files define _FILE_OFFSET_BITS to 64 to enable large file support
# this might break the precompiled header usage for GCC as it requires the macro set to be consistent
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND MR_PCH)
  add_compile_definitions(_FILE_OFFSET_BITS=64)
endif()
