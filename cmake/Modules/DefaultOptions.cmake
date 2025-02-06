# this file must be included BEFORE the `project' command because the compiler flags are crucial for the platform detection

IF(DEFINED ENV{MR_USE_CPP_23} AND "$ENV{MR_USE_CPP_23}" STREQUAL "ON")
  message("MR_USE_CPP_23 variable is deprecated; consider setting MR_CXX_STANDARD to 23")
  set(MR_CXX_STANDARD 23 CACHE STRING "Version of the C++ standard used to compile the project")
ELSE()
  set(MR_CXX_STANDARD 20 CACHE STRING "Version of the C++ standard used to compile the project")
ENDIF()
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
