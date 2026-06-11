# mr_win32_enable_mimalloc_redirect(<target>)
#
# Enable mimalloc's transparent allocator redirect for a Windows executable.
# Two conditions must hold for the redirect to engage at runtime:
#   1. /INCLUDE:mi_version pulls mimalloc.dll into the EXE's import table.
#   2. mimalloc.dll must be the FIRST entry in the EXE's import table so
#      mimalloc-redirect.dll loads before ucrtbase.dll. We achieve this by
#      prepending mimalloc to LINK_LIBRARIES; otherwise libs that transitively
#      pull ucrtbase (cpr, gmock, libcurl, ...) come earlier in the link order,
#      ucrtbase initializes first, and mimalloc-redirect bails with "standard
#      malloc is _not_ redirected!".
#
# This is a Windows-only, EXE-only mechanism; guard the call with if(WIN32).
# The MSBuild equivalent lives at MeshLib/source/MimallocRedirect.props.
# Linking the imported mimalloc target lets CMake resolve the version-specific
# import lib name (mimalloc.lib for 2.x, mimalloc.dll.lib for 3.x).
# (Emscripten links mimalloc via -sMALLOC=mimalloc instead, see
# ConfigureEmscripten.cmake.)
function(mr_win32_enable_mimalloc_redirect target)
  find_package(mimalloc CONFIG REQUIRED)
  get_target_property(_existing_libs ${target} LINK_LIBRARIES)
  if(_existing_libs)
    set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc ${_existing_libs})
  else()
    set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc)
  endif()
  target_link_options(${target} PRIVATE "/INCLUDE:mi_version")
endfunction()
