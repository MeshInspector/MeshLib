#include "MRBitSet.h"
#include "MRGTest.h"

namespace MR
{

bool BitSet::any() const
{
    for ( const auto b : blocks_ )
        if ( b )
            return true;
    return false;
}

void BitSet::zeroUnusedBits()
{
    assert ( num_blocks() == calcNumBlocks( numBits_ ) );
    if ( auto extraBits = numBits_ % bitsPerBlock )
        blocks_.back() &= bitMask( extraBits ) - 1;
}

BitSet & BitSet::set( IndexType n, size_type len )
{
    return rangeOp( n, len,
        []( Block ){ return ~Block{}; },
        []( Block b, size_t firstBit, size_t lastBit ){ return b | bitMask( firstBit, lastBit ); } );
}

BitSet & BitSet::reset( IndexType n, size_type len )
{
    return rangeOp( n, len,
        []( Block ){ return Block{}; },
        []( Block b, size_t firstBit, size_t lastBit ){ return b & ~bitMask( firstBit, lastBit ); } );
}

BitSet & BitSet::set()
{
    blocks_.clear();
    blocks_.resize( calcNumBlocks( numBits_ ), ~Block{} );
    zeroUnusedBits();
    return * this;
}

BitSet & BitSet::reset()
{
    blocks_.clear();
    blocks_.resize( calcNumBlocks( numBits_ ), Block{} );
    return * this;
}

BitSet & BitSet::flip()
{
    for ( auto & b : blocks_ )
        b = ~b;
    zeroUnusedBits();
    return * this;
}

void BitSet::resize( size_type numBits, bool fillValue )
{
    ...
}

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
    const auto endBlock = std::min( num_blocks(), rhs.num_blocks() );
    for ( size_type i = 0; i < endBlock; ++i )
        m_bits[i] &= ~rhs.m_bits[i];
    return *this;
}

BitSet & BitSet::subtract( const BitSet & b, int bShiftInBlocks )
{
    const auto beginBlock = std::max( 0, bShiftInBlocks );
    const auto endBlock = std::clamp( b.num_blocks() + bShiftInBlocks, size_t(0), num_blocks() );
    for ( size_type i = beginBlock; i < endBlock; ++i )
        m_bits[i] &= ~b.m_bits[i - bShiftInBlocks];
    return *this;
}

bool operator == ( const BitSet & a, const BitSet & b )
{
    if ( a.size() == b.size() )
        return static_cast<const BitSet::base &>( a ) == static_cast<const BitSet::base &>( b );

    auto aBlocksNum = a.num_blocks();
    auto bBlocksNum = b.num_blocks();
    auto minBlocksNum = std::min( aBlocksNum, bBlocksNum );
    for ( size_t i = 0; i < std::min( aBlocksNum, bBlocksNum ); ++i )
        if ( a.bits()[i] != b.bits()[i] )
            return false;
    const auto& maxBitSet = aBlocksNum > bBlocksNum ? a : b;
    for ( size_t i = minBlocksNum; i < maxBitSet.num_blocks(); ++i )
        if ( maxBitSet.bits()[i] != 0 )
            return false;
    return true;
}

BitSet::IndexType BitSet::find_last() const
{
    if ( !any() )
        return npos;
    for ( IndexType i = size(); i-- >= 1; )
    {
        if ( test( i ) )
            return i;
    }
    return npos;
}

size_t BitSet::nthSetBit( size_t n ) const
{
    for ( auto b : *this )
        if ( n-- == 0 )
            return b;
    return npos;
}

bool BitSet::is_subset_of( const BitSet& a ) const
{
    // base implementation does not support bitsets of different sizes
    const auto commonBlocks = std::min( num_blocks(), a.num_blocks() );
    for ( size_type i = 0; i < commonBlocks; ++i )
        if ( m_bits[i] & ~a.m_bits[i] )
            return false;
    // this is subset of (a) if consider common bits only

    return size() <= a.size() // this has no more bits than (a)
        || find_next( a.size() - 1 ) > size(); // or all additional bits of this are off
}

} //namespace MR
