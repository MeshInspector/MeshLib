#include "MRBitSet.h"
#include "MRMacros.h"
#include <bit>
#include <iostream>

#if defined( __x86_64__ ) || defined( _M_X64 )
#define MR_POPCNT_DISPATCH
#if defined( _MSC_VER ) && !defined( __clang__ )
#include <intrin.h>
#define MR_TARGET_POPCNT
#else
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define MR_TARGET_POPCNT __attribute__(( target( "popcnt" ) ))
#endif
#endif

namespace MR
{

namespace
{

using Block = BitSet::block_type;
using Counts = BitSet::Counts;

template <bool hw>
MR_FORCE_INLINE int popcount( Block b ) noexcept
{
#if defined( _MSC_VER ) && !defined( __clang__ ) && defined( MR_POPCNT_DISPATCH )
    if constexpr ( hw )
        return int( __popcnt64( b ) );
    else
#endif
    return std::popcount( b );
}

template <bool hw>
MR_FORCE_INLINE size_t countT( const Block* blocks, size_t numBlocks ) noexcept
{
    size_t res = 0;
    for ( size_t i = 0; i < numBlocks; ++i )
        res += popcount<hw>( blocks[i] );
    return res;
}

template <bool hw>
MR_FORCE_INLINE Counts countBitsAndBlocksT( const Block* blocks, size_t numBlocks, size_t maxBits ) noexcept
{
    size_t bits = 0, nonEmptyBlocks = 0; // not Counts members, which MSVC keeps in memory
    for ( size_t i = 0; i < numBlocks; ++i )
    {
        const auto b = blocks[i];
        if ( !b )
            continue;
        ++nonEmptyBlocks;
        bits += popcount<hw>( b );
        if ( bits > maxBits )
            break;
    }
    return { bits, nonEmptyBlocks };
}

size_t countSw( const Block* blocks, size_t numBlocks ) noexcept { return countT<false>( blocks, numBlocks ); }
Counts countBitsAndBlocksSw( const Block* blocks, size_t numBlocks, size_t maxBits ) noexcept { return countBitsAndBlocksT<false>( blocks, numBlocks, maxBits ); }

// constant-initialized, so the portable versions serve calls from static initializers that run before selectPopcnt
auto countImpl = &countSw;
auto countBitsAndBlocksImpl = &countBitsAndBlocksSw;

#ifdef MR_POPCNT_DISPATCH
MR_TARGET_POPCNT size_t countHw( const Block* blocks, size_t numBlocks ) noexcept { return countT<true>( blocks, numBlocks ); }
MR_TARGET_POPCNT Counts countBitsAndBlocksHw( const Block* blocks, size_t numBlocks, size_t maxBits ) noexcept { return countBitsAndBlocksT<true>( blocks, numBlocks, maxBits ); }

bool cpuHasPopcnt()
{
#ifdef _MSC_VER
    int info[4];
    __cpuid( info, 1 );
    return info[2] & ( 1 << 23 );
#else
    unsigned a, b, c, d;
    return __get_cpuid( 1, &a, &b, &c, &d ) && ( c & bit_POPCNT );
#endif
}

[[maybe_unused]] const bool selectPopcnt = []
{
    if ( !cpuHasPopcnt() )
        return false;
    countImpl = &countHw;
    countBitsAndBlocksImpl = &countBitsAndBlocksHw;
    return true;
}();
#endif

} // anonymous namespace

size_t countSetBits( const BitSet::block_type* blocks, size_t numBlocks ) noexcept
{
    return countImpl( blocks, numBlocks );
}

bool BitSet::any() const
{
    for ( const auto b : blocks_ )
        if ( b )
            return true;
    return false;
}

bool BitSet::all() const
{
    auto lastBlock = blockIndex_( numBits_ );
    if ( auto lastBit = bitIndex_( numBits_ ); lastBit > 0 && blocks_[lastBlock] != bitMask_( lastBit ) - 1 )
        return false;
    for ( size_t i = 0; i < lastBlock; ++i )
        if ( blocks_[i] != ~block_type{} )
            return false;
    return true;
}

auto BitSet::count() const noexcept -> size_type
{
    return countImpl( blocks_.data(), blocks_.size() );
}

auto BitSet::countBitsAndBlocks( size_type maxBits ) const noexcept -> Counts
{
    return countBitsAndBlocksImpl( blocks_.data(), blocks_.size(), maxBits );
}

void BitSet::setUnusedBits_()
{
    assert ( num_blocks() == calcNumBlocks_( numBits_ ) );
    if ( auto lastBit = bitIndex_( numBits_ ) )
        blocks_.back() |= ~( bitMask_( lastBit ) - 1 );
}

void BitSet::resetUnusedBits_()
{
    assert ( num_blocks() == calcNumBlocks_( numBits_ ) );
    if ( auto lastBit = bitIndex_( numBits_ ) )
        blocks_.back() &= bitMask_( lastBit ) - 1;
}

template<class FullBlock, class PartialBlock>
BitSet & BitSet::rangeOp_( IndexType n, size_type len, FullBlock&& f, PartialBlock&& p )
{
    assert( n <= size() );
    assert( n + len <= size() );
    if ( len == 0 )
        return *this;

    const auto firstBlock = blockIndex_( n );
    const auto firstBit = bitIndex_( n );

    const auto lastBlock = blockIndex_( n + len );
    const auto lastBit = bitIndex_( n + len );

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

BitSet & BitSet::set( IndexType n, size_type len, bool val )
{
    if ( !val )
        return reset( n, len );
    return rangeOp_( n, len,
        []( block_type ){ return ~block_type{}; },
        []( block_type b, size_t firstBit, size_t lastBit ){ return b | bitMask_( firstBit, lastBit ); } );
}

BitSet & BitSet::reset( IndexType n, size_type len )
{
    return rangeOp_( n, len,
        []( block_type ){ return block_type{}; },
        []( block_type b, size_t firstBit, size_t lastBit ){ return b & ~bitMask_( firstBit, lastBit ); } );
}

BitSet & BitSet::flip( IndexType n, size_type len )
{
    return rangeOp_( n, len,
        []( block_type b ){ return ~b; },
        []( block_type b, size_t firstBit, size_t lastBit ){ return b ^ bitMask_( firstBit, lastBit ); } );
}

BitSet & BitSet::set()
{
    blocks_.clear();
    blocks_.resize( calcNumBlocks_( numBits_ ), ~block_type{} );
    resetUnusedBits_();
    return * this;
}

BitSet & BitSet::reset()
{
    blocks_.clear();
    blocks_.resize( calcNumBlocks_( numBits_ ), block_type{} );
    return * this;
}

BitSet & BitSet::flip()
{
    for ( auto & b : blocks_ )
        b = ~b;
    resetUnusedBits_();
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
        setUnusedBits_();
        blocks_.resize( calcNumBlocks_( numBits ), ~block_type{} );
    }
    else
        blocks_.resize( calcNumBlocks_( numBits ), block_type{} );
    numBits_ = numBits;
    resetUnusedBits_();
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
    if ( blocks_.empty() )
        return npos;

    auto lastBit = bitIndex_( numBits_ );
    auto block = blocks_.back();
    if ( lastBit != 0 )
        block &= ( ( block_type( 1 ) << lastBit ) - 1 );
    if ( block != block_type( 0 ) )
        return blocks_.size() * bits_per_block - std::countl_zero( block ) - 1;
    for ( int i = int( blocks_.size() ) - 2; i >= 0; --i )
    {
        if ( blocks_[i] == block_type( 0 ) )
            continue;
        return ( i + 1 ) * bits_per_block - std::countl_zero( blocks_[i] ) - 1;
    }
    return npos;
}

BitSet::IndexType BitSet::find_last_not_set() const
{
    if ( blocks_.empty() )
        return npos;

    auto lastBit = bitIndex_( numBits_ );
    auto block = blocks_.back();
    if ( lastBit != 0 )
        block |= ~( ( block_type( 1 ) << lastBit ) - 1 );
    if ( block != ~block_type( 0 ) )
        return blocks_.size() * bits_per_block - std::countl_one( block ) - 1;
    for ( int i = int( blocks_.size() ) - 2; i >= 0; --i )
    {
        if ( blocks_[i] == ~block_type( 0 ) )
            continue;
        return ( i + 1 ) * bits_per_block - std::countl_one( blocks_[i] ) - 1;
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

    auto blockId = blockIndex_( n );
    const auto firstBit = bitIndex_( n );
    auto block0 = blocks_[blockId];
    block0 &= ~( ( block_type( 1 ) << firstBit ) - 1 ); // zero bits before firstBit
    if ( auto c = std::countr_zero( block0 ); c < bits_per_block )
        return blockId * bits_per_block + c;
    for ( ++blockId; blockId < blocks_.size(); ++blockId )
        if ( auto c = std::countr_zero( blocks_[blockId] ); c < bits_per_block )
            return blockId * bits_per_block + c;
    return npos;
}

auto BitSet::findNonSetBitAfter_( IndexType n ) const -> IndexType
{
    if ( n >= size() )
        return npos;

    auto blockId = blockIndex_( n );
    const auto firstBit = bitIndex_( n );
    auto block0 = blocks_[blockId];
    block0 |= ( ( block_type( 1 ) << firstBit ) - 1 ); // one bits before firstBit
    if ( auto c = std::countr_one( block0 ); c < bits_per_block )
        return blockId * bits_per_block + c;
    for ( ++blockId; blockId < blocks_.size(); ++blockId )
        if ( auto c = std::countr_one( blocks_[blockId] ); c < bits_per_block )
            return blockId * bits_per_block + c;
    return npos;
}

std::ostream& operator<<( std::ostream& s, const BitSet & bs )
{
    auto i = bs.size();
    while ( i > 0 )
    {
        --i;
        s.put( bs.test( i ) ? '1' : '0' );
    }
    return s;
}

std::istream& operator>>( std::istream& s, BitSet & bs )
{
    bs.clear();
    while ( true )
    {
        auto c = s.peek();
        if ( c != '0' && c != '1' )
            break;
        (void)s.get();
        bs.push_back( c == '1' );
    }
    bs.reverse();
    return s;
}

HashMap<size_t, size_t> makeHashMapWithSeqNums( const BitSet & bs )
{
    HashMap<size_t, size_t> res;
    int n = 0;
    for ( auto v : bs )
        res[v] = n++;
    return res;
}

} //namespace MR
