# MeshLib libraries allocate some containers (e.g. BitSet::Blocks) explicitly with mimalloc and never override malloc,
# so they can be loaded into any process (Python, .NET, Java); executables override malloc with the same mimalloc,
# see mr_enable_mimalloc.
# Without it (MRMESH_NO_MIMALLOC) MeshLib libraries use std::allocator: by default in Emscripten builds, which get mimalloc
# from -sMALLOC=mimalloc (see ConfigureEmscripten.cmake), and in macOS Homebrew builds, where Homebrew's mimalloc.h
# of any version comes first in the include path.
if(EMSCRIPTEN OR (APPLE AND MESHLIB_USE_HOMEBREW))
  set(_mr_no_mimalloc_default ON)
else()
  set(_mr_no_mimalloc_default OFF)
endif()
option(MRMESH_NO_MIMALLOC "MeshLib libraries use std::allocator instead of mimalloc" ${_mr_no_mimalloc_default})

# mr_find_mimalloc()
#
# Provides the `mimalloc` target for MeshLib libraries (unless MRMESH_NO_MIMALLOC).
# Windows: vcpkg's mimalloc[override], whose redirect engages only in mr_enable_mimalloc executables.
# macOS and Linux: MeshLib's own non-override mimalloc, named MRMimalloc: an override one crashes a plain Python loading it late.
set(MR_MIMALLOC_PACKAGE_LOCK ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/cpm/package-lock.cmake)
function(mr_find_mimalloc)
  if(WIN32)
    find_package(mimalloc CONFIG REQUIRED)
  else()
    include(${MR_MIMALLOC_PACKAGE_LOCK})
    include(FetchContent)
    # without MeshLib's -Werror -pedantic-errors
    set(CMAKE_C_FLAGS "")
    set(MI_OVERRIDE OFF)
    if(NOT APPLE)
      # dlopen-safe thread-locals; on macOS 13 this TLS model aborts in dyld at launch
      set(MI_LOCAL_DYNAMIC_TLS ON)
    endif()
    set(MI_BUILD_SHARED ON)
    set(MI_BUILD_STATIC OFF)
    set(MI_BUILD_OBJECT OFF)
    set(MI_BUILD_TESTS OFF)
    set(MI_INSTALL_TOPLEVEL ON)
    # the allocator of the whole process in mr_enable_mimalloc executables: release-built in every build type
    set(CMAKE_BUILD_TYPE Release)
    FetchContent_Declare(mimalloc ${MESHLIB_PACKAGE_mimalloc})
    # not the headers of another mimalloc version, e.g. Homebrew's from the directory's include_directories
    get_directory_property(_include_dirs INCLUDE_DIRECTORIES)
    set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "")
    FetchContent_MakeAvailable(mimalloc)
    set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${_include_dirs}")
    set_target_properties(mimalloc PROPERTIES OUTPUT_NAME MRMimalloc)
    # consumers compile mimalloc.h with their warnings, e.g. -Wdeprecated-copy in mi_stl_allocator
    get_target_property(_mimalloc_includes mimalloc INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(mimalloc PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_mimalloc_includes}")
    install(TARGETS mimalloc LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}" RUNTIME DESTINATION "${MR_BIN_DIR}")

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
  elseif(MRMESH_NO_MIMALLOC)
    # e.g. macOS Homebrew: its shared mimalloc loaded at launch replaces malloc for the whole process;
    # MeshLib libraries use std::allocator there
    find_package(mimalloc 3.5 CONFIG REQUIRED)
    target_link_libraries(${target} PRIVATE mimalloc)
  else()
    # malloc and new forward to the libraries' MRMimalloc, so the process has one mimalloc;
    # first in the link order, before libstdc++ and libc on Linux; dyld interposes it on macOS
    get_target_property(_existing_libs ${target} LINK_LIBRARIES)
    if(_existing_libs)
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES MRMimallocOverride ${_existing_libs})
    else()
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES MRMimallocOverride)
    endif()
  endif()
endfunction()
