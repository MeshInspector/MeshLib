#include "MRBitSet.h"

#include "MRMesh/MRBitSet.h"

using namespace MR;

const uint64_t* mrBitSetBlocks( const MRBitSet* bs )
{
    return reinterpret_cast<const BitSet*>( bs )->m_bits.data();
}

size_t mrBitSetSize( const MRBitSet* bs )
{
    return reinterpret_cast<const BitSet*>( bs )->size();
}

void mrBitSetFree( MRBitSet* bs )
{
    delete reinterpret_cast<const BitSet*>( bs );
}
