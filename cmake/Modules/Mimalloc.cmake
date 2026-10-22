# MeshLib libraries allocate explicitly with mimalloc (e.g. BitSet's mi_stl_allocator) and never override malloc,
# so they can be loaded into any process (Python, .NET, Java). An application that wants mimalloc for all its
# allocations sets MR_MIMALLOC_OVERRIDE=ON and calls mr_enable_mimalloc on its executable.
# Emscripten builds get mimalloc from -sMALLOC=mimalloc instead, see ConfigureEmscripten.cmake.
option(MR_MIMALLOC_OVERRIDE "Link MeshLib with a malloc-overriding mimalloc for mr_enable_mimalloc executables" OFF)

# mr_find_mimalloc()
#
# Provides the `mimalloc` target; there must be only one mimalloc in a process.
# Windows: vcpkg's mimalloc[override], whose redirect engages only in mr_enable_mimalloc executables.
# macOS and Linux: the dependencies' non-override build, or with MR_MIMALLOC_OVERRIDE an override build made here
# (named mimalloc-override, so it does not collide with the dependencies' one in packages).
set(MR_MIMALLOC_PACKAGE_LOCK ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/cpm/package-lock.cmake)
function(mr_find_mimalloc)
  if(TARGET mimalloc)
  elseif(WIN32)
    find_package(mimalloc CONFIG REQUIRED)
  elseif(NOT MR_MIMALLOC_OVERRIDE)
    find_package(mimalloc 3.5 CONFIG REQUIRED HINTS ${MESHLIB_THIRDPARTY_ROOT_DIR}/lib/cmake/mimalloc)
  else()
    include(${MR_MIMALLOC_PACKAGE_LOCK})
    include(FetchContent)
    # without MeshLib's -Werror -pedantic-errors
    set(CMAKE_C_FLAGS "")
    set(MI_OVERRIDE ON)
    set(MI_BUILD_SHARED ON)
    set(MI_BUILD_STATIC OFF)
    set(MI_BUILD_OBJECT OFF)
    set(MI_BUILD_TESTS OFF)
    set(MI_INSTALL_TOPLEVEL ON)
    FetchContent_Declare(mimalloc ${MESHLIB_PACKAGE_mimalloc})
    FetchContent_MakeAvailable(mimalloc)
    set_target_properties(mimalloc PROPERTIES OUTPUT_NAME mimalloc-override)
    install(TARGETS mimalloc LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}" RUNTIME DESTINATION "${MR_BIN_DIR}")
  endif()
endfunction()

# mr_enable_mimalloc(<target>)
#
# Replace the system allocator of an executable with mimalloc for the whole process.
function(mr_enable_mimalloc target)
  if(EMSCRIPTEN)
    return()
  endif()
  mr_find_mimalloc()
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
    get_target_property(_existing_libs ${target} LINK_LIBRARIES)
    if(_existing_libs)
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc ${_existing_libs})
    else()
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES mimalloc)
    endif()
    target_link_options(${target} PRIVATE "/INCLUDE:mi_version")
  else()
    if(NOT MR_MIMALLOC_OVERRIDE)
      message(FATAL_ERROR "mr_enable_mimalloc(${target}) requires MR_MIMALLOC_OVERRIDE=ON")
    endif()
    # the shared override build loaded at launch replaces malloc for the whole process
    target_link_libraries(${target} PRIVATE mimalloc)
  endif()
endfunction()
