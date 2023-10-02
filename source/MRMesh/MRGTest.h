#pragma once
#include "MRStreamOperators.h"

#ifdef MRMESH_NO_GTEST

#undef TEST
#undef ASSERT_EQ
#undef EXPECT_EQ
#undef ASSERT_NE
#undef EXPECT_NE
#undef ASSERT_LE
#undef EXPECT_LE
#undef ASSERT_GE
#undef EXPECT_GE
#undef ASSERT_GT
#undef EXPECT_GT
#undef ASSERT_TRUE
#undef EXPECT_TRUE
#undef ASSERT_FALSE
#undef EXPECT_FALSE
#undef ASSERT_NEAR
#undef EXPECT_NEAR

#define TEST( a, b ) [[maybe_unused]] static void a##b()
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

#else

#include <gtest/gtest.h>

#endif
