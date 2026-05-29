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

  find_package(mimalloc CONFIG REQUIRED)
  if(WIN32)
    # mimalloc.dll must be the FIRST PE import (prepend) so its redirect beats
    # ucrtbase; /INCLUDE:mi_version forces the import.
    get_target_property(_existing_libs ${target} LINK_LIBRARIES)
    if(_existing_libs)
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc ${_existing_libs})
    else()
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc)
    endif()
    target_link_options(${target} PRIVATE "/INCLUDE:mi_version")
  else()
    # minimal: just link it; refine per-platform only if the test shows non-engagement
    target_link_libraries(${target} PRIVATE mimalloc)
  endif()

  target_compile_definitions(${target} PRIVATE MR_MIMALLOC_ENABLED)
endfunction()
