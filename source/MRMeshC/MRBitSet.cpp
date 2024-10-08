#include "MRBitSet.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRBitSet.h"

using namespace MR;

REGISTER_AUTO_CAST( BitSet )

const uint64_t* mrBitSetBlocks( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.m_bits.data();
}

size_t mrBitSetBlocksNum( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.m_bits.size();
}

size_t mrBitSetSize( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.size();
}

void mrBitSetFree( MRBitSet* bs_ )
{
    ARG_PTR( bs );
    delete bs;
}

MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs_ )
{
    auto&& fbs = *cast_to<FaceBitSet>( fbs_ );
    return cast_to<MRFaceBitSet>( new FaceBitSet( fbs ) );
}

void mrFaceBitSetFree( MRFaceBitSet* fbs_ )
{
    auto&& fbs = cast_to<FaceBitSet>( fbs_ );
    delete fbs;
}

bool mrBitSetEq( const MRBitSet* a_, const MRBitSet* b_ )
{
    ARG( a ); ARG( b );
    return a == b;
}

MRFaceBitSet* mrFaceBitSetNew( void )
{
    return cast_to<MRFaceBitSet>( new FaceBitSet );
}

size_t mrBitSetCount( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.count();
}

size_t mrBitSetFindFirst( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.find_first();
}

size_t mrBitSetFindLast( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.find_last();
}

void mrBitSetResize( MRBitSet* bs_, size_t size, bool value )
{
    ARG( bs );
    bs.resize( size, value );
}

void mrBitSetAutoResizeSet( MRBitSet* bs_, size_t pos, bool value )
{
    ARG( bs );
    bs.autoResizeSet( pos, value );
}

MRBitSet* mrBitSetSub( const MRBitSet* a_, const MRBitSet* b_ )
{
    ARG( a ); ARG( b );
    RETURN_NEW( a - b );
}
