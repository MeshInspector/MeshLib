#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <time.h>

#define RUN_TEST( func )                                         \
    printf( "%s ...\n", #func );                                 \
    {                                                            \
        time_t ts = time( NULL );                                \
        func();                                                  \
        time_t duration_s = time( NULL ) - ts;                   \
        printf( "%s done (~ %d s)\n", #func, (int)duration_s );  \
    }

#define TEST_ASSERT( ... )  \
    if ( !( __VA_ARGS__ ) ) \
    {                       \
        fprintf( stderr, "%s:%d: check failed: %s\n", __func__, __LINE__, ( #__VA_ARGS__ ) ); \
        abort();            \
    }

#define TEST_ASSERT_INT_EQUAL( val, exp ) \
    if ( !( val == exp ) )                \
    {                                     \
        fprintf( stderr, "%s:%d: check failed: expected %d, got %d\n", __func__, __LINE__, exp, val ); \
        abort();                          \
    }

#define TEST_ASSERT_FLOAT_EQUAL_APPROX( val, exp, eps ) \
    if ( !( fabs( val - exp ) <= eps ) )                \
    {                                                   \
        fprintf( stderr, "%s:%d: check failed: expected %f (delta %f), got %f (delta %f)\n", __func__, __LINE__, exp, eps, val, fabs( val - exp ) ); \
        abort();                                        \
    }
