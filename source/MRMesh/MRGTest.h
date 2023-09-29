#pragma once
#include "MRStreamOperators.h"

#ifdef MRMESH_NO_GTEST

#define MR_TEST( a, b ) [[maybe_unused]] static void a##b()

#else

#include <gtest/gtest.h>
#define MR_TEST( a, b ) TEST( a, b )

#endif

