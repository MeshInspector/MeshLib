#include "MRBitSet.h"
#include "MRGTest.h"

namespace MR
{

BitSet & BitSet::operator &= ( const BitSet & rhs )
{
    resize( std::min( size(), rhs.size() ) );
    for ( size_type i = 0; i < num_blocks(); ++i )
        m_bits[i] &= rhs.m_bits[i];
    return *this;
}

BitSet & BitSet::operator |= ( const BitSet & rhs )
{
    resize( std::max( size(), rhs.size() ) );
    for ( size_type i = 0; i < rhs.num_blocks(); ++i )
        m_bits[i] |= rhs.m_bits[i];
    return *this;
}

BitSet & BitSet::operator ^= ( const BitSet & rhs )
{
    resize( std::max( size(), rhs.size() ) );
    for ( size_type i = 0; i < rhs.num_blocks(); ++i )
        m_bits[i] ^= rhs.m_bits[i];
    return *this;
}

BitSet & BitSet::operator -= ( const BitSet & rhs )
{
    resize( std::max( size(), rhs.size() ) );
    for ( size_type i = 0; i < rhs.num_blocks(); ++i )
        m_bits[i] &= ~rhs.m_bits[i];
    return *this;
}

BitSet::IndexType BitSet::find_last() const
{
    if ( !any() )
        return base::npos;
    for ( IndexType i = size(); i-- >= 1; )
    {
        if ( test( i ) )
            return i;
    }
    return base::npos;
}

TEST(MRMesh, BitSet) 
{
    BitSet bs0(4);
    bs0.set(0);
    bs0.set(2);

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
}

TEST(MRMesh, TaggedBitSet) 
{
    VertBitSet bs0( 3 );
    bs0.set( VertId( 0 ) );
    bs0.set( VertId( 2 ) );

    VertBitSet bs1( 4 );
    bs1.set( VertId( 1 ) );
    bs1.set( VertId( 2 ) );

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
