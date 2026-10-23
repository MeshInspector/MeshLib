# MeshLib libraries allocate some containers (e.g. BitSet::Blocks) explicitly with mimalloc and never override malloc,
# so they can be loaded into any process (Python, .NET, Java).
# An application overriding malloc in its executables with mr_enable_mimalloc sets MR_MIMALLOC_OVERRIDE=ON;
# MeshLib libraries then use the standard allocators and do not link mimalloc, and MRTest and MeshViewer override malloc too.
# On Linux the libraries keep MRMimalloc, and the executables' override forwards to it (MRMimallocOverride).
# Emscripten builds get mimalloc from -sMALLOC=mimalloc instead, see ConfigureEmscripten.cmake.
option(MR_MIMALLOC_OVERRIDE "Executables override malloc with mimalloc instead of MeshLib libraries using it explicitly" OFF)
if(EMSCRIPTEN OR (MR_MIMALLOC_OVERRIDE AND (WIN32 OR APPLE)))
  set(MR_MIMALLOC_IN_LIBRARIES OFF)
else()
  set(MR_MIMALLOC_IN_LIBRARIES ON)
endif()

# mr_find_mimalloc()
#
# Provides the `mimalloc` target for MeshLib libraries (MR_MIMALLOC_IN_LIBRARIES).
# Windows: vcpkg's mimalloc[override], whose redirect engages only in mr_enable_mimalloc executables.
# macOS and Linux: the dependencies' mimalloc overrides malloc, which crashes a plain Python loading it late,
# so MeshLib builds its own non-override copy of the same version, named MRMimalloc.
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

    if(MR_MIMALLOC_OVERRIDE)
      # malloc, new and the rest of glibc's allocation functions, for mr_enable_mimalloc
      set(_override_dir ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../source/MRMimallocOverride)
      add_library(MRMimallocOverride SHARED ${_override_dir}/MRMimallocOverride.c ${_override_dir}/MRMimallocNewDelete.cpp)
      target_link_libraries(MRMimallocOverride PRIVATE mimalloc)
      set_source_files_properties(${_override_dir}/MRMimallocOverride.c PROPERTIES COMPILE_OPTIONS -fno-builtin)
      install(TARGETS MRMimallocOverride LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}")
    endif()
  endif()
endfunction()

# mr_enable_mimalloc(<target>)
#
# Replace the system allocator of an executable with mimalloc for the whole process.
function(mr_enable_mimalloc target)
  if(NOT WIN32 AND NOT MR_MIMALLOC_OVERRIDE)
    message(FATAL_ERROR "mr_enable_mimalloc(${target}) requires MR_MIMALLOC_OVERRIDE=ON")
  endif()
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
  elseif(NOT APPLE AND NOT EMSCRIPTEN)
    # Linux: malloc and new forward to the libraries' MRMimalloc, so the process has one mimalloc;
    # first in the link order, before libstdc++ and libc
    get_target_property(_existing_libs ${target} LINK_LIBRARIES)
    if(_existing_libs)
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES MRMimallocOverride ${_existing_libs})
    else()
      set_property(TARGET ${target} PROPERTY LINK_LIBRARIES MRMimallocOverride)
    endif()
  elseif(APPLE)
    # macOS: the dependencies' shared mimalloc loaded at launch replaces malloc for the whole process;
    # >= 3.5 (thirdparty/vcpkg/ports-unix, thirdparty/cpm on Ubuntu):
    # a dynamic 3.4.3 aborts at launch on macOS 13 (dyld thread-local bootstrap during libSystem init)
    find_package(mimalloc 3.5 CONFIG REQUIRED)
    target_link_libraries(${target} PRIVATE mimalloc)
  endif()
endfunction()
