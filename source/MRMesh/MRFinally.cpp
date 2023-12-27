#include "MRFinally.h"

#include "MRGTest.h"

TEST( MRFinally, Normal )
{
    bool x = false;

    {
        MR_FINALLY{ x = true; };
        ASSERT_EQ( x, false );
    }
    ASSERT_EQ( x, true );
}

TEST( MRFinally, OnSuccess )
{
    bool x = false;

    {
        MR_FINALLY_ON_SUCCESS{ x = true; };
        ASSERT_EQ( x, false );
    }
    ASSERT_EQ( x, true );
}

TEST( MRFinally, OnThrow )
{
    bool x = false;

    {
        MR_FINALLY_ON_THROW{ x = true; };
        ASSERT_EQ( x, false );
    }
    ASSERT_EQ( x, false );
}

// We disable exception tests on Emscripten, because there you can have disabled exception catching that can't be directly detected with any macros.
#if defined(__cpp_exceptions) && !defined(__EMSCRIPTEN__)

TEST( MRFinally, Normal_Exception )
{
    bool x = false;

    try
    {
        MR_FINALLY{ x = true; };
        ASSERT_EQ( x, false );
        throw 42;
    }
    catch ( int ) {}
    ASSERT_EQ( x, true );
}

TEST( MRFinally, OnSuccess_Exception )
{
    bool x = false;

    try
    {
        MR_FINALLY_ON_SUCCESS{ x = true; };
        ASSERT_EQ( x, false );
        throw 42;
    }
    catch ( int ) {}
    ASSERT_EQ( x, false );
}

TEST( MRFinally, OnThrow_Exception )
{
    bool x = false;

    try
    {
        MR_FINALLY_ON_THROW{ x = true; };
        ASSERT_EQ( x, false );
        throw 42;
    }
    catch ( int )
    {
    }
    ASSERT_EQ( x, true );
}

#endif
