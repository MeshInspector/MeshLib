#pragma once

#if defined( _MSC_VER )
// no config file support yet
#elif __has_include( "config_cmake.h" )
#include "config_cmake.h"
#endif
