#pragma once

// MRVoxels/config.h
// This comment is added to fix GCC builds.

#if defined( MR_USE_CMAKE_CONFIGURE_FILE ) && __has_include( "config_cmake.h" )
#include "config_cmake.h"
#endif
