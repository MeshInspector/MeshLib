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
    # Linux: mimalloc's single object file linked into the executable overrides glibc's malloc for the
    # whole process; a separate build keeps mimalloc's install rules out of the packages
    set(_mimalloc_dir ${CMAKE_BINARY_DIR}/mimalloc)
    if(NOT TARGET mimalloc_build)
      include(ExternalProject)
      ExternalProject_Add(mimalloc_build
        URL https://github.com/microsoft/mimalloc/archive/refs/tags/v3.4.3.tar.gz
        URL_HASH SHA256=7b043c26b64dde66344d144fbcac3664858a4eb32197793ea2a18feb32741ca1
        INSTALL_DIR ${_mimalloc_dir}
        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
          -DCMAKE_INSTALL_LIBDIR=lib
          -DMI_INSTALL_TOPLEVEL=ON
          -DMI_USE_CXX=ON
          -DMI_BUILD_SHARED=OFF
          -DMI_BUILD_TESTS=OFF
        BUILD_BYPRODUCTS ${_mimalloc_dir}/lib/mimalloc.o
      )
    endif()
    add_dependencies(${target} mimalloc_build)
    target_link_libraries(${target} PRIVATE ${_mimalloc_dir}/lib/mimalloc.o)
  endif()
endfunction()
