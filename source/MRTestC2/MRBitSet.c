#include "TestMacros.h"

#include "MRBitSet.h"

#include <MRCMesh/MRBitSet.h>

void testBitSet( void )
{
    MR_BitSet* a = MR_BitSet_Construct( 10, false );
    TEST_ASSERT( MR_BitSet_size( a ) == 10 )

    MR_BitSet_set_2( a, 5, true );
    TEST_ASSERT( MR_BitSet_test( a, 5 ) )
    TEST_ASSERT( !MR_BitSet_test( a, 4 ) )
    TEST_ASSERT( MR_BitSet_find_last( a ) == 5 )

    MR_BitSet* b = MR_BitSet_Construct( 0, false );
    MR_BitSet_autoResizeSet_2( b, 6, &(bool){true} );
    TEST_ASSERT( MR_BitSet_size( b ) == 7 )

    MR_BitSet_resize( a, 10, false );
    MR_BitSet_set_2( a, 5, true );
    MR_BitSet_set_2( a, 6, true );
    MR_BitSet_resize( b, 10, false );
    MR_BitSet_set_2( b, 6, true );
    MR_BitSet* c = MR_sub_MR_BitSet( a, b );
    TEST_ASSERT( MR_BitSet_test( c, 5 ) )
    TEST_ASSERT( !MR_BitSet_test( c, 6 ) )

    MR_BitSet_Destroy( c );
    MR_BitSet_Destroy( b );
    MR_BitSet_Destroy( a );
}
