# mr_enable_mimalloc(<target>)
#
# Replace the system allocator of an executable with mimalloc for the whole process.
# Emscripten builds get mimalloc from -sMALLOC=mimalloc instead, see ConfigureEmscripten.cmake.
function(mr_enable_mimalloc target)
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
  elseif(APPLE)
    # static mimalloc force-loaded: it replaces the default malloc zone for the whole process
    find_package(mimalloc CONFIG REQUIRED)
    target_link_options(${target} PRIVATE "LINKER:-force_load,$<TARGET_FILE:mimalloc-static>")
  elseif(NOT EMSCRIPTEN)
    # Linux: the shared mimalloc overrides glibc's malloc for the whole process, and MeshLib libraries
    # linking the same target share this one copy; thirdparty/cpm builds vcpkg's version for Ubuntu
    find_package(mimalloc 3.4 EXACT CONFIG REQUIRED)
    target_link_libraries(${target} PRIVATE mimalloc)
  endif()
endfunction()
