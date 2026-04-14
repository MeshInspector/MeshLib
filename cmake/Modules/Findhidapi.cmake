# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#a-sample-find-module

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_hidapi QUIET hidapi-hidraw)
endif()

find_path(hidapi_INCLUDE_DIR
  NAMES hidapi/hidapi.h
  HINTS ${PC_hidapi_INCLUDE_DIRS}
)
find_library(hidapi_LIBRARY
  NAMES hidapi-hidraw
  HINTS ${PC_hidapi_LIBRARY_DIRS}
)

if(hidapi_INCLUDE_DIR)
  file(READ "${hidapi_INCLUDE_DIR}/hidapi/hidapi.h" hidapi_VERSION_FILE)
  string(REGEX MATCH "HID_API_VERSION_MAJOR ([0-9]+)" _ ${hidapi_VERSION_FILE})
  set(hidapi_VERSION_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "HID_API_VERSION_MINOR ([0-9]+)" _ ${hidapi_VERSION_FILE})
  set(hidapi_VERSION_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "HID_API_VERSION_PATCH ([0-9]+)" _ ${hidapi_VERSION_FILE})
  set(hidapi_VERSION_PATCH ${CMAKE_MATCH_1})
  set(hidapi_VERSION "${hidapi_VERSION_MAJOR}.${hidapi_VERSION_MINOR}.${hidapi_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hidapi
  REQUIRED_VARS hidapi_INCLUDE_DIR hidapi_LIBRARY
  VERSION_VAR hidapi_VERSION
)
mark_as_advanced(
  hidapi_INCLUDE_DIR
  hidapi_LIBRARY
)

if(hidapi_FOUND AND NOT TARGET hidapi::hidapi)
  add_library(hidapi::hidapi UNKNOWN IMPORTED)
  set_target_properties(hidapi::hidapi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${hidapi_INCLUDE_DIR}
    IMPORTED_LOCATION ${hidapi_LIBRARY}
    VERSION ${hidapi_VERSION}
  )
endif()
