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
    # the mimalloc dylib loaded at launch interposes malloc and replaces the default malloc zone for the whole process;
    # it needs mimalloc >= 3.5 (ports-macos): a dynamic 3.4.3 aborts at launch on macOS 13 (dyld thread-local bootstrap during libSystem init)
    find_package(mimalloc 3.5 CONFIG REQUIRED)
    target_link_libraries(${target} PRIVATE mimalloc)
  elseif(NOT EMSCRIPTEN)
    # Linux: the shared mimalloc overrides glibc's malloc for the whole process, and MeshLib libraries
    # linking the same target share this one copy; thirdparty/cpm builds vcpkg's version for Ubuntu
    find_package(mimalloc 3.4 EXACT CONFIG REQUIRED)
    target_link_libraries(${target} PRIVATE mimalloc)
  endif()
endfunction()
