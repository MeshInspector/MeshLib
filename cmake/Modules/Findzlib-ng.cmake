# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#a-sample-find-module

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_zlib_ng QUIET zlib-ng)
endif()

find_path(zlib-ng_INCLUDE_DIR
  NAMES zlib-ng.h
  HINTS ${PC_zlib_ng_INCLUDE_DIRS}
)
find_library(zlib-ng_LIBRARY
  NAMES z-ng zlib-ng
  HINTS ${PC_zlib_ng_LIBRARY_DIRS}
)

if(zlib-ng_INCLUDE_DIR)
  file(READ "${zlib-ng_INCLUDE_DIR}/zlib-ng.h" zlib-ng_VERSION_FILE)
  string(REGEX MATCH "ZLIBNG_VERSION \"([0-9\\.]+)\"" _ "${zlib-ng_VERSION_FILE}")
  set(zlib-ng_VERSION "${CMAKE_MATCH_1}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(zlib-ng
  REQUIRED_VARS zlib-ng_INCLUDE_DIR zlib-ng_LIBRARY
  VERSION_VAR zlib-ng_VERSION
)
mark_as_advanced(
  zlib-ng_INCLUDE_DIR
  zlib-ng_LIBRARY
)

if(zlib-ng_FOUND AND NOT TARGET zlib-ng::zlib)
  add_library(zlib-ng::zlib UNKNOWN IMPORTED)
  set_target_properties(zlib-ng::zlib PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${zlib-ng_INCLUDE_DIR}"
    IMPORTED_LOCATION "${zlib-ng_LIBRARY}"
    VERSION "${zlib-ng_VERSION}"
  )
endif()
