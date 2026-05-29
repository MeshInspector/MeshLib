# mr_enable_mimalloc(<target>) - enable the mimalloc allocator for an EXE target.
# EXE-only; call unconditionally (it branches internally). Gated by MESHLIB_USE_MIMALLOC.
# Verified by MRMesh.MimallocRedirectActive, which keys off the MR_MIMALLOC_ENABLED
# define this sets. MSBuild twin: MimallocRedirect.props.
function(mr_enable_mimalloc target)
  if(NOT MESHLIB_USE_MIMALLOC)
    return()
  endif()

  if(EMSCRIPTEN OR MR_EMSCRIPTEN)
    target_link_options(${target} PRIVATE "-sMALLOC=mimalloc")
    return()
  endif()

  if(WIN32)
    # mimalloc.dll must be the FIRST PE import (prepend) so its redirect beats
    # ucrtbase; /INCLUDE:mi_version forces the import.
    find_package(mimalloc CONFIG REQUIRED)
    get_target_property(_existing_libs ${target} LINK_LIBRARIES)
    if(_existing_libs)
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc ${_existing_libs})
    else()
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc)
    endif()
    target_link_options(${target} PRIVATE "/INCLUDE:mi_version")
  else()
    # Linux/macOS: prefer the CMake config (ubuntu24, brew, vcpkg); Ubuntu 22.04's
    # libmimalloc-dev ships no config, so fall back to finding the library directly.
    find_package(mimalloc CONFIG QUIET)
    if(TARGET mimalloc)
      target_link_libraries(${target} PRIVATE mimalloc)
    else()
      find_library(MR_MIMALLOC_LIBRARY NAMES mimalloc REQUIRED)
      target_link_libraries(${target} PRIVATE ${MR_MIMALLOC_LIBRARY})
    endif()
  endif()

  target_compile_definitions(${target} PRIVATE MR_MIMALLOC_ENABLED)
endfunction()
