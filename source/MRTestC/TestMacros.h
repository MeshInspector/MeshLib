#pragma once

#include <stdio.h>
#include <stdlib.h>

#define EXPECT( ... ) \
    if ( !( __VA_ARGS__ ) ) \
    {                 \
        fprintf( stderr, "check failed: %s", ( #__VA_ARGS__ ) ); \
        abort();      \
    }
