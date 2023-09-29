#pragma once
#include "MRStreamOperators.h"

#ifdef MRMESH_NO_GTEST

#define MR_TEST( a, b ) [[maybe_unused]] static void a##b()

#ifndef ASSERT_EQ

#define ASSERT_EQ( a, b ) (void)( a == b );
#define EXPECT_EQ( a, b ) (void)( a == b );
#define ASSERT_NE( a, b ) (void)( a != b );
#define EXPECT_NE( a, b ) (void)( a != b );
#define ASSERT_LE( a, b ) (void)( a <= b );
#define EXPECT_LE( a, b ) (void)( a <= b );
#define ASSERT_GE( a, b ) (void)( a >= b );
#define EXPECT_GE( a, b ) (void)( a >= b );
#define ASSERT_GT( a, b ) (void)( a > b );
#define EXPECT_GT( a, b ) (void)( a > b );
#define ASSERT_TRUE( a ) (void)( a );
#define EXPECT_TRUE( a ) (void)( a );
#define ASSERT_FALSE( a ) (void)( !(a) );
#define EXPECT_FALSE( a ) (void)( !(a) );
#define ASSERT_NEAR( a, b, e ) (void)( ( a - b ) <= e );
#define EXPECT_NEAR( a, b, e ) (void)( ( a - b ) <= e );

#endif // ASSERT_EQ

#else

#include <gtest/gtest.h>
#define MR_TEST( a, b ) TEST( a, b )

#endif
