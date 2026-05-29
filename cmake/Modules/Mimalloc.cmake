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
  elseif(APPLE)
    # macOS: static-link mimalloc via -force_load. brew's DYNAMIC mimalloc 3.x aborts
    # at launch on macOS <= 13: an allocation during libSystem init hits mi_thread_init,
    # which touches a thread-local before dyld can bootstrap it -> dyld abort. Static-
    # linking ties mimalloc's init to this exe's own initializers (after libSystem),
    # avoiding that early path.
    find_package(mimalloc CONFIG QUIET)
    if(TARGET mimalloc-static)
      target_link_options(${target} PRIVATE "LINKER:-force_load,$<TARGET_FILE:mimalloc-static>")
    else()
      set(_mi_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
      set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
      find_library(MR_MIMALLOC_STATIC NAMES mimalloc REQUIRED)
      set(CMAKE_FIND_LIBRARY_SUFFIXES ${_mi_saved_suffixes})
      target_link_options(${target} PRIVATE "LINKER:-force_load,${MR_MIMALLOC_STATIC}")
    endif()
  else()
    # Linux: dynamic link engages the override (ubuntu24 via CONFIG; ubuntu22's
    # libmimalloc-dev ships no config, so fall back to find_library).
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
