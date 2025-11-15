# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#a-sample-find-module

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_Freetype QUIET freetype2)
endif()

find_path(Freetype_INCLUDE_DIR
  NAMES freetype/freetype.h
  HINTS ${PC_Freetype_INCLUDE_DIRS}
  PATH_SUFFIXES freetype2
)
find_library(Freetype_LIBRARY
  NAMES freetype
  HINTS ${PC_Freetype_LIBRARY_DIRS}
)

if(Freetype_INCLUDE_DIR)
  file(READ "${Freetype_INCLUDE_DIR}/freetype/freetype.h" Freetype_VERSION_FILE)
  string(REGEX MATCH "FREETYPE_MAJOR  ([0-9]+)" _ ${Freetype_VERSION_FILE})
  set(Freetype_VERSION_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "FREETYPE_MINOR  ([0-9]+)" _ ${Freetype_VERSION_FILE})
  set(Freetype_VERSION_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "FREETYPE_PATCH  ([0-9]+)" _ ${Freetype_VERSION_FILE})
  set(Freetype_VERSION_PATCH ${CMAKE_MATCH_1})
  set(Freetype_VERSION "${Freetype_VERSION_MAJOR}.${Freetype_VERSION_MINOR}.${Freetype_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Freetype
  REQUIRED_VARS Freetype_INCLUDE_DIR Freetype_LIBRARY
  VERSION_VAR Freetype_VERSION
)
mark_as_advanced(
  Freetype_INCLUDE_DIR
  Freetype_LIBRARY
)

if(Freetype_FOUND AND NOT TARGET Freetype::Freetype)
  add_library(Freetype::Freetype UNKNOWN IMPORTED)
  set_target_properties(Freetype::Freetype PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${Freetype_INCLUDE_DIR}
    IMPORTED_LOCATION ${Freetype_LIBRARY}
    VERSION ${Freetype_VERSION}
  )
endif()
