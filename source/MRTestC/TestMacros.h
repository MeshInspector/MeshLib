#pragma once

#include "TestFunctions.h"

#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

#define RUN_TEST( func )                      \
    printf( "%s ...\n", #func );              \
    {                                         \
        struct timespec ts1 = timespec_now(); \
        func();                               \
        struct timespec ts2 = timespec_now(); \
        struct timespec duration = timespec_get_duration( &ts1, &ts2 );          \
        printf( "%s done (%.3f s)\n", #func, timespec_to_seconds( &duration ) ); \
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
