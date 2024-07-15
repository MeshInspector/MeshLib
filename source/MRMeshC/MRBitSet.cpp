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

MRFaceBitSet* mrFaceBitSetNew()
{
    return reinterpret_cast<MRFaceBitSet*>( new FaceBitSet );
}

size_t mrBitSetCount( const MRBitSet* bs_ )
{
    const auto& bs = *reinterpret_cast<const BitSet*>( bs_ );

    return bs.count();
}
