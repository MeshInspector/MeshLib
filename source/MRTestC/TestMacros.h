#pragma once

#include <stdio.h>
#include <stdlib.h>

#define TEST_ASSERT( ... ) if ( !( __VA_ARGS__ ) ) { fprintf( stderr, "%s:%d: check failed: %s\n", __func__, __LINE__, ( #__VA_ARGS__ ) ); abort(); }
