#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST(MRMesh, BitSet)
{
    BitSet bs0(4);
    bs0.set(0);
    bs0.set(2);

    EXPECT_EQ( bs0.nthSetBit( 0 ), 0 );
    EXPECT_EQ( bs0.nthSetBit( 1 ), 2 );
    EXPECT_EQ( bs0.nthSetBit( 2 ), BitSet::npos );

    BitSet bs1(3);
    bs1.set(1);
    bs1.set(2);

    EXPECT_TRUE(  end( bs1 ) == std::find_if( begin( bs1 ), end( bs1 ), []( size_t i ) { return i == 0; } ) );
    EXPECT_FALSE( end( bs1 ) == std::find_if( begin( bs1 ), end( bs1 ), []( size_t i ) { return i == 1; } ) );

    EXPECT_EQ( BitSet( bs0 & bs1 ).count(), 1 );
    EXPECT_EQ( BitSet( bs0 | bs1 ).count(), 3 );
    EXPECT_EQ( BitSet( bs0 - bs1 ).count(), 1 );
    EXPECT_EQ( BitSet( bs1 - bs0 ).count(), 1 );
    EXPECT_EQ( BitSet( bs0 ^ bs1 ).count(), 2 );

    EXPECT_EQ( BitSet( BitSet( bs0 ) &= bs1 ).count(), 1 );
    EXPECT_EQ( BitSet( BitSet( bs0 ) |= bs1 ).count(), 3 );
    EXPECT_EQ( BitSet( BitSet( bs0 ) -= bs1 ).count(), 1 );
    EXPECT_EQ( BitSet( BitSet( bs1 ) -= bs0 ).count(), 1 );
    EXPECT_EQ( BitSet( BitSet( bs0 ) ^= bs1 ).count(), 2 );

    EXPECT_EQ( bs0.find_last(), size_t( 2 ) );
    BitSet bs2( 5 );
    EXPECT_EQ( bs2.find_last(), size_t( -1 ) );

    EXPECT_FALSE( bs1.is_subset_of( bs0 ) );
    EXPECT_FALSE( bs0.is_subset_of( bs1 ) );
    BitSet bs3 = bs0;
    EXPECT_TRUE( bs3.is_subset_of( bs0 ) );
    bs3.resize( 5 );
    EXPECT_TRUE( bs3.is_subset_of( bs0 ) );
    EXPECT_TRUE( bs0.is_subset_of( bs3 ) );
    bs3.set( 4 );
    EXPECT_FALSE( bs3.is_subset_of( bs0 ) );
    EXPECT_TRUE( bs0.is_subset_of( bs3 ) );
}

TEST(MRMesh, TaggedBitSet)
{
    VertBitSet bs0( 3 );
    bs0.set( VertId( 0 ) );
    bs0.set( VertId( 2 ) );

    VertBitSet bs1( 4 );
    bs1.set( VertId( 1 ) );
    bs1.set( VertId( 2 ) );

    EXPECT_EQ( bs1.nthSetBit( 0 ), 1_v );
    EXPECT_EQ( bs1.nthSetBit( 1 ), 2_v );
    EXPECT_EQ( bs1.nthSetBit( 2 ), VertId{} );

    EXPECT_EQ( VertBitSet( bs0 & bs1 ).count(), 1 );
    EXPECT_EQ( VertBitSet( bs0 | bs1 ).count(), 3 );
    EXPECT_EQ( VertBitSet( bs0 - bs1 ).count(), 1 );
    EXPECT_EQ( VertBitSet( bs1 - bs0 ).count(), 1 );
    EXPECT_EQ( VertBitSet( bs0 ^ bs1 ).count(), 2 );

    EXPECT_EQ( VertBitSet( VertBitSet( bs0 ) &= bs1 ).count(), 1 );
    EXPECT_EQ( VertBitSet( VertBitSet( bs0 ) |= bs1 ).count(), 3 );
    EXPECT_EQ( VertBitSet( VertBitSet( bs0 ) -= bs1 ).count(), 1 );
    EXPECT_EQ( VertBitSet( VertBitSet( bs1 ) -= bs0 ).count(), 1 );
    EXPECT_EQ( VertBitSet( VertBitSet( bs0 ) ^= bs1 ).count(), 2 );
}

} //namespace MR
