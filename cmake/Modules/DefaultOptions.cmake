# this file must be included BEFORE the `project' command because the compiler flags are crucial for the platform detection

IF(DEFINED ENV{MR_USE_CPP_23} AND "$ENV{MR_USE_CPP_23}" STREQUAL "ON")
  message("MR_USE_CPP_23 variable is deprecated; consider setting MR_CXX_STANDARD to 23")
  set(MR_CXX_STANDARD 23 CACHE STRING "Version of the C++ standard used to compile the project")
ELSE()
  set(MR_CXX_STANDARD 20 CACHE STRING "Version of the C++ standard used to compile the project")
ENDIF()
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(MR_USE_CMAKE_CONFIGURE_FILE)

# MSVC debug information format
IF(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
ENDIF()

if(APPLE)
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
  endif()
  if(NOT CMAKE_OSX_SYSROOT AND NOT DEFINED ENV{SDKROOT})
    set(CMAKE_OSX_SYSROOT "macosx" CACHE STRING "macOS SDK")
  endif()
endif()

if(MR_EMSCRIPTEN)
  include(DefaultEmscriptenOptions)
endif()
