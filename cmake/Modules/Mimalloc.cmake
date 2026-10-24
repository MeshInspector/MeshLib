# MeshLib libraries allocate some containers (e.g. BitSet::Blocks) explicitly with mimalloc and never override malloc,
# so they can be loaded into any process (Python, .NET, Java).
# An application overriding malloc in its executables with mr_enable_mimalloc sets MR_MIMALLOC_OVERRIDE=ON,
# and MRTest and MeshViewer override malloc too; the override uses the same mimalloc as the libraries.
# Emscripten builds get mimalloc from -sMALLOC=mimalloc instead, see ConfigureEmscripten.cmake.
# macOS Homebrew builds use std::allocator: Homebrew's mimalloc overrides malloc.
option(MR_MIMALLOC_OVERRIDE "Executables override malloc with the mimalloc of MeshLib libraries" OFF)
if(EMSCRIPTEN OR (APPLE AND MESHLIB_USE_HOMEBREW))
  set(MR_STD_ALLOCATOR ON)
else()
  set(MR_STD_ALLOCATOR OFF)
endif()

# mr_find_mimalloc()
#
# Provides the `mimalloc` target for MeshLib libraries (unless MR_STD_ALLOCATOR).
# Windows: vcpkg's mimalloc[override], whose redirect engages only in mr_enable_mimalloc executables.
# macOS and Linux: the dependencies' mimalloc 3.5 without the malloc override (thirdparty/cpm, thirdparty/vcpkg/ports-unix):
# an override one crashes a plain Python loading it late, and a dynamic 3.4.3 aborts at launch on macOS 13.
function(mr_find_mimalloc)
  if(WIN32)
    find_package(mimalloc CONFIG REQUIRED)
  else()
    find_package(mimalloc 3.5 CONFIG REQUIRED)
  endif()

  if(MR_MIMALLOC_OVERRIDE AND NOT WIN32)
    # malloc, new and the rest of the system allocation functions, for mr_enable_mimalloc
    if(APPLE)
      set(_override_source MRMimallocOverrideMacOS.cpp)
    else()
      set(_override_source MRMimallocOverride.cpp)
    endif()
    add_library(MRMimallocOverride SHARED ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../source/MRMimallocOverride/${_override_source})
    target_link_libraries(MRMimallocOverride PRIVATE mimalloc)
    target_compile_options(MRMimallocOverride PRIVATE -fno-builtin)
    install(TARGETS MRMimallocOverride LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}")
  endif()
endfunction()

# mr_enable_mimalloc(<target>)
#
# Replace the system allocator of an executable with mimalloc for the whole process.
function(mr_enable_mimalloc target)
  if(NOT WIN32 AND NOT MR_MIMALLOC_OVERRIDE)
    message(FATAL_ERROR "mr_enable_mimalloc(${target}) requires MR_MIMALLOC_OVERRIDE=ON")
  endif()
  if(WIN32)
    # Two conditions must hold for mimalloc's redirect to engage at runtime:
    #   1. /INCLUDE:mi_version pulls mimalloc.dll into the EXE's import table.
    #   2. mimalloc.dll must be the FIRST entry in the EXE's import table so
    #      mimalloc-redirect.dll loads before ucrtbase.dll. We achieve this by
    #      prepending mimalloc to LINK_LIBRARIES; otherwise libs that transitively
    #      pull ucrtbase (cpr, gmock, libcurl, ...) come earlier in the link order,
    #      ucrtbase initializes first, and mimalloc-redirect bails with "standard
    #      malloc is _not_ redirected!".
    # The MSBuild equivalent lives at MeshLib/source/MimallocRedirect.props.
    # Linking the imported mimalloc target lets CMake resolve the version-specific
    # import lib name (mimalloc.lib for 2.x, mimalloc.dll.lib for 3.x).
    find_package(mimalloc CONFIG REQUIRED)
    get_target_property(_existing_libs ${target} LINK_LIBRARIES)
    if(_existing_libs)
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc ${_existing_libs})
    else()
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc)
    endif()
    target_link_options(${target} PRIVATE "/INCLUDE:mi_version")
  elseif(EMSCRIPTEN)
    # -sMALLOC=mimalloc, see ConfigureEmscripten.cmake
  elseif(MR_STD_ALLOCATOR)
    # macOS Homebrew: its shared mimalloc loaded at launch replaces malloc for the whole process;
    # MeshLib libraries use std::allocator there
    find_package(mimalloc 3.5 CONFIG REQUIRED)
    target_link_libraries(${target} PRIVATE mimalloc)
  else()
    # malloc and new forward to the libraries' mimalloc, so the process has one mimalloc;
    # first in the link order, before libstdc++ and libc on Linux; dyld interposes it on macOS
    get_target_property(_existing_libs ${target} LINK_LIBRARIES)
    if(_existing_libs)
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES MRMimallocOverride ${_existing_libs})
    else()
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES MRMimallocOverride)
    endif()
  endif()
endfunction()
