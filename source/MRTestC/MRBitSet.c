#include "TestMacros.h"

#include "MRBitSet.h"

#include <MRMeshC/MRBitSet.h>

void testBitSet( void )
{
    MRBitSet* a = mrBitSetNew( 10, false );
    TEST_ASSERT( mrBitSetSize( a ) == 10 )

    mrBitSetSet( a, 5, true );
    TEST_ASSERT( mrBitSetTest( a, 5 ) )
    TEST_ASSERT( !mrBitSetTest( a, 4 ) )
    TEST_ASSERT( mrBitSetFindLast( a ) == 5 )

    MRBitSet* b = mrBitSetNew( 0, false );
    mrBitSetAutoResizeSet( b, 6, true );
    TEST_ASSERT( mrBitSetSize( b ) == 7 )

    mrBitSetResize( a, 10, false );
    mrBitSetSet( a, 5, true );
    mrBitSetSet( a, 6, true );
    mrBitSetResize( b, 10, false );
    mrBitSetSet( b, 6, true );
    MRBitSet* c = mrBitSetSub( a, b );
    TEST_ASSERT( mrBitSetTest( c, 5 ) )
    TEST_ASSERT( !mrBitSetTest( c, 6 ) )

    mrBitSetFree( c );
    mrBitSetFree( b );
    mrBitSetFree( a );
}
