# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#a-sample-find-module

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_JsonCpp QUIET jsoncpp)
endif()

find_path(JsonCpp_INCLUDE_DIR
  NAMES json/json.h json/version.h
  HINTS ${PC_JsonCpp_INCLUDE_DIRS}
)
find_library(JsonCpp_LIBRARY
  NAMES jsoncpp
  HINTS ${PC_JsonCpp_LIBRARY_DIRS}
)

if(JsonCpp_INCLUDE_DIR)
  file(READ "${JsonCpp_INCLUDE_DIR}/json/version.h" JsonCpp_VERSION_FILE)
  string(REGEX MATCH "JSONCPP_VERSION_STRING \"([0-9\.]+)\"" _ ${JsonCpp_VERSION_FILE})
  set(JsonCpp_VERSION ${CMAKE_MATCH_1})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JsonCpp
  REQUIRED_VARS JsonCpp_INCLUDE_DIR JsonCpp_LIBRARY
  VERSION_VAR JsonCpp_VERSION
)
mark_as_advanced(
  JsonCpp_INCLUDE_DIR
  JsonCpp_LIBRARY
)

if(JsonCpp_FOUND AND NOT TARGET JsonCpp::JsonCpp)
  add_library(JsonCpp::JsonCpp UNKNOWN IMPORTED)
  set_target_properties(JsonCpp::JsonCpp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${JsonCpp_INCLUDE_DIR}
    IMPORTED_LOCATION ${JsonCpp_LIBRARY}
    VERSION ${JsonCpp_VERSION}
  )
endif()
