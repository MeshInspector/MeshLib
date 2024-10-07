#include "MRBitSet.h"

#include "MRMesh/MRBitSet.h"

using namespace MR;

const uint64_t* mrBitSetBlocks( const MRBitSet* bs )
{
    return reinterpret_cast<const BitSet*>( bs )->m_bits.data();
}

size_t mrBitSetBlocksNum( const MRBitSet* bs )
{
    return reinterpret_cast<const BitSet*>( bs )->m_bits.size();
}

size_t mrBitSetSize( const MRBitSet* bs )
{
    return reinterpret_cast<const BitSet*>( bs )->size();
}

void mrBitSetFree( MRBitSet* bs )
{
    delete reinterpret_cast<BitSet*>( bs );
}

MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs )
{
    auto* res = new FaceBitSet( *reinterpret_cast<const FaceBitSet*>( fbs ) );
    return reinterpret_cast<MRFaceBitSet*>( res );
}

void mrFaceBitSetFree( MRFaceBitSet* fbs )
{
    delete reinterpret_cast<FaceBitSet*>( fbs );
}

bool mrBitSetEq( const MRBitSet* a, const MRBitSet* b )
{
    return *reinterpret_cast<const BitSet*>( a ) == *reinterpret_cast<const BitSet*>( b );
}

MRFaceBitSet* mrFaceBitSetNew( void )
{
    return reinterpret_cast<MRFaceBitSet*>( new FaceBitSet );
}

size_t mrBitSetCount( const MRBitSet* bs_ )
{
    const auto& bs = *reinterpret_cast<const BitSet*>( bs_ );

    return bs.count();
}

size_t mrBitSetFindFirst( const MRBitSet* bs_ )
{
    const auto& bs = *reinterpret_cast<const BitSet*>( bs_ );

    return bs.find_first();
}

size_t mrBitSetFindLast( const MRBitSet* bs_ )
{
    const auto& bs = *reinterpret_cast<const BitSet*>( bs_ );

    return bs.find_last();
}

void mrBitSetResize( MRBitSet* bs_, size_t size, bool value )
{
    auto& bs = *reinterpret_cast<BitSet*>( bs_ );

    bs.resize( size, value );
}

void mrBitSetAutoResizeSet( MRBitSet* bs_, size_t pos, bool value )
{
    auto& bs = *reinterpret_cast<BitSet*>( bs_ );

    bs.autoResizeSet( pos, value );
}

MRBitSet* mrBitSetSub( const MRBitSet* a_, const MRBitSet* b_ )
{
    const auto& a = *reinterpret_cast<const BitSet*>( a_ );
    const auto& b = *reinterpret_cast<const BitSet*>( b_ );

    auto result = a - b;

    return reinterpret_cast<MRBitSet*>( new BitSet( std::move( result ) ) );
}
