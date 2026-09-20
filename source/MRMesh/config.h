#pragma once

// MRMesh/config.h
// This comment is added to make the config header files unique.
// More info: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52566

#if defined( MR_USE_CMAKE_CONFIGURE_FILE ) && __has_include( "config_cmake.h" )
#include "config_cmake.h"
#endif

// written by scripts/make_install_folder.py, present in distribution packages only
#if __has_include( "config_dist.h" )
#include "config_dist.h"
#endif
