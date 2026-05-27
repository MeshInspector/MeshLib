# mr_enable_mimalloc_redirect(<target>)
#
# Enable mimalloc's transparent allocator redirect for a Windows executable:
# link the dynamic mimalloc and force-keep the mi_version symbol so that
# mimalloc.dll is loaded early enough for mimalloc-redirect.dll to take over the
# CRT allocator process-wide (covering all allocations, including those made
# inside MRMesh and the other MeshLib DLLs).
#
# The redirect is an executable-only, Windows-only mechanism, so this is a no-op
# on every other platform (Emscripten links mimalloc via -sMALLOC=mimalloc, see
# ConfigureEmscripten.cmake). Linking the imported mimalloc target lets CMake
# resolve the version-specific import library name for us.
function(mr_enable_mimalloc_redirect target)
  if(NOT WIN32)
    return()
  endif()
  find_package(mimalloc CONFIG REQUIRED)
  target_link_libraries(${target} PRIVATE mimalloc)
  target_link_options(${target} PRIVATE "/INCLUDE:mi_version")
endfunction()
