#include "MRBitSet.h"
#include "MRGTest.h"
#include <bit>

namespace MR
{

bool BitSet::any() const
{
    for ( const auto b : blocks_ )
        if ( b )
            return true;
    return false;
}

bool BitSet::all() const
{
    auto lastBlock = blockIndex( numBits_ );
    if ( auto lastBit = bitIndex( numBits_ ); lastBit > 0 && blocks_[lastBlock] != bitMask( lastBit ) - 1 )
        return false;
    for ( size_t i = 0; i < lastBlock; ++i )
        if ( blocks_[i] != ~block_type{} )
            return false;
    return true;
}

auto BitSet::count() const noexcept -> size_type
{
    size_type res = 0;
    for ( const auto b : blocks_ )
        res += std::popcount( b );
    return res;
}

void BitSet::setUnusedBits()
{
    assert ( num_blocks() == calcNumBlocks( numBits_ ) );
    if ( auto lastBit = bitIndex( numBits_ ) )
        blocks_.back() |= ~( bitMask( lastBit ) - 1 );
}

void BitSet::resetUnusedBits()
{
    assert ( num_blocks() == calcNumBlocks( numBits_ ) );
    if ( auto lastBit = bitIndex( numBits_ ) )
        blocks_.back() &= bitMask( lastBit ) - 1;
}

template<class FullBlock, class PartialBlock>
BitSet & BitSet::rangeOp( IndexType n, size_type len, FullBlock&& f, PartialBlock&& p )
{
    assert( n <= size() );
    assert( n + len <= size() );
    if ( len == 0 )
        return *this;

    const auto firstBlock = blockIndex( n );
    const auto firstBit = bitIndex( n );

    const auto lastBlock = blockIndex( n + len );
    const auto lastBit = bitIndex( n + len );

    if ( firstBlock == lastBlock )
    {
        blocks_[firstBlock] = p( blocks_[firstBlock], firstBit, lastBit );
        return *this;
    }

    if ( firstBit > 0 )
        blocks_[firstBlock] = p( blocks_[firstBlock], firstBit, bits_per_block );

    if ( lastBit > 0 )
        blocks_[lastBlock] = p( blocks_[lastBlock], 0, lastBit );

    const auto firstFullBlock = ( firstBit == 0 ) ? firstBlock : firstBlock + 1;
    for ( auto i = firstFullBlock; i < lastBlock; ++i )
        blocks_[i] = f( blocks_[i] );

    return *this;
}

BitSet & BitSet::set( IndexType n, size_type len )
{
    return rangeOp( n, len,
        []( block_type ){ return ~block_type{}; },
        []( block_type b, size_t firstBit, size_t lastBit ){ return b | bitMask( firstBit, lastBit ); } );
}

BitSet & BitSet::reset( IndexType n, size_type len )
{
    return rangeOp( n, len,
        []( block_type ){ return block_type{}; },
        []( block_type b, size_t firstBit, size_t lastBit ){ return b & ~bitMask( firstBit, lastBit ); } );
}

BitSet & BitSet::set()
{
    blocks_.clear();
    blocks_.resize( calcNumBlocks( numBits_ ), ~block_type{} );
    resetUnusedBits();
    return * this;
}

BitSet & BitSet::reset()
{
    blocks_.clear();
    blocks_.resize( calcNumBlocks( numBits_ ), block_type{} );
    return * this;
}

BitSet & BitSet::flip()
{
    for ( auto & b : blocks_ )
        b = ~b;
    resetUnusedBits();
    return * this;
}

void BitSet::reverse()
{
    if ( size() <= 1 )
        return;
    IndexType i = 0, j = size() - 1;
    while( i < j )
    {
        bool ti = test( i );
        bool tj = test( j );
        set( i, tj );
        set( j, ti );
        ++i;
        --j;
    }
}

void BitSet::resize( size_type numBits, bool fillValue )
{
    if ( fillValue )
    {
        setUnusedBits();
        blocks_.resize( calcNumBlocks( numBits ), ~block_type{} );
    }
    else
        blocks_.resize( calcNumBlocks( numBits ), block_type{} );
    numBits_ = numBits;
    resetUnusedBits();
}

BitSet & BitSet::operator &= ( const BitSet & rhs )
{
    resize( std::min( size(), rhs.size() ) );
    for ( size_type i = 0; i < num_blocks(); ++i )
        blocks_[i] &= rhs.blocks_[i];
    return *this;
}

BitSet & BitSet::operator |= ( const BitSet & rhs )
{
    resize( std::max( size(), rhs.size() ) );
    for ( size_type i = 0; i < rhs.num_blocks(); ++i )
        blocks_[i] |= rhs.blocks_[i];
    return *this;
}

BitSet & BitSet::operator ^= ( const BitSet & rhs )
{
    resize( std::max( size(), rhs.size() ) );
    for ( size_type i = 0; i < rhs.num_blocks(); ++i )
        blocks_[i] ^= rhs.blocks_[i];
    return *this;
}

BitSet & BitSet::operator -= ( const BitSet & rhs )
{
    const auto endBlock = std::min( num_blocks(), rhs.num_blocks() );
    for ( size_type i = 0; i < endBlock; ++i )
        blocks_[i] &= ~rhs.blocks_[i];
    return *this;
}

BitSet & BitSet::subtract( const BitSet & b, int bShiftInBlocks )
{
    const auto beginBlock = std::max( 0, bShiftInBlocks );
    const auto endBlock = std::clamp( b.num_blocks() + bShiftInBlocks, size_t(0), num_blocks() );
    for ( size_type i = beginBlock; i < endBlock; ++i )
        blocks_[i] &= ~b.blocks_[i - bShiftInBlocks];
    return *this;
}

bool operator == ( const BitSet & a, const BitSet & b )
{
    if ( a.size() == b.size() )
        return a.bits() == b.bits();

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
    const auto commonBlocks = std::min( num_blocks(), a.num_blocks() );
    for ( size_type i = 0; i < commonBlocks; ++i )
        if ( blocks_[i] & ~a.blocks_[i] )
            return false;
    // this is subset of (a) if consider common bits only

    return size() <= a.size() // this has no more bits than (a)
        || find_next( a.size() - 1 ) > size(); // or all additional bits of this are off
}

bool BitSet::intersects( const BitSet& a ) const
{
    const auto commonBlocks = std::min( num_blocks(), a.num_blocks() );
    for ( size_type i = 0; i < commonBlocks; ++i )
        if ( blocks_[i] & a.blocks_[i] )
            return true;

    return false;
}

auto BitSet::findSetBitAfter_( IndexType n ) const -> IndexType
{
    if ( n >= size() )
        return npos;

    auto blockId = blockIndex( n );
    const auto firstBit = bitIndex( n );
    auto block0 = blocks_[blockId];
    block0 &= ~( ( block_type( 1 ) << firstBit ) - 1 ); // zero bits before firstBit
    if ( auto c = std::countr_zero( block0 ); c < bits_per_block )
        return blockId * bits_per_block + c;
    for ( ++blockId; blockId < blocks_.size(); ++blockId )
        if ( auto c = std::countr_zero( blocks_[blockId] ); c < bits_per_block )
            return blockId * bits_per_block + c;
    return npos;
}

} //namespace MR
