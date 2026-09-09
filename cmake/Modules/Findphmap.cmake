# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#a-sample-find-module

find_path(phmap_INCLUDE_DIR
  NAMES parallel_hashmap/phmap.h
  HINTS
    # within MeshLib's own build
    ${MESHLIB_THIRDPARTY_ROOT_DIR}/include
    # from an installed MeshLib, where meshlib-config.cmake defines this before find_dependency
    ${MESHLIB_THIRDPARTY_INCLUDE_DIR}
)

if(phmap_INCLUDE_DIR)
  file(READ "${phmap_INCLUDE_DIR}/parallel_hashmap/phmap_config.h" phmap_CONFIG_FILE)
  string(REGEX MATCH "PHMAP_VERSION_MAJOR ([0-9]+)" _ ${phmap_CONFIG_FILE})
  set(phmap_VERSION_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "PHMAP_VERSION_MINOR ([0-9]+)" _ ${phmap_CONFIG_FILE})
  set(phmap_VERSION_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "PHMAP_VERSION_PATCH ([0-9]+)" _ ${phmap_CONFIG_FILE})
  set(phmap_VERSION_PATCH ${CMAKE_MATCH_1})
  set(phmap_VERSION "${phmap_VERSION_MAJOR}.${phmap_VERSION_MINOR}.${phmap_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(phmap
  REQUIRED_VARS phmap_INCLUDE_DIR
  VERSION_VAR phmap_VERSION
)
mark_as_advanced(
  phmap_INCLUDE_DIR
)

if(phmap_FOUND AND NOT TARGET phmap::phmap)
  add_library(phmap::phmap INTERFACE IMPORTED)
  set_target_properties(phmap::phmap PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${phmap_INCLUDE_DIR}"
    VERSION ${phmap_VERSION}
  )
endif()
