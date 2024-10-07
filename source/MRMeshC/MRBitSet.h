#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// gets read-only access to the underlying blocks of a bitset
MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

/// gets count of the underlying blocks of a bitset
MRMESHC_API size_t mrBitSetBlocksNum( const MRBitSet* bs );

/// gets total length of a bitset
MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

/// returns the number of bits in this bitset that are set
MRMESHC_API size_t mrBitSetCount( const MRBitSet* bs );

/// checks if two bitsets are equal (have the same length and identical bit values)
MRMESHC_API bool mrBitSetEq( const MRBitSet* a, const MRBitSet* b );

/// ...
MRMESHC_API size_t mrBitSetFindFirst( const MRBitSet* bs );

/// ...
MRMESHC_API size_t mrBitSetFindLast( const MRBitSet* bs );

/// ...
MRMESHC_API void mrBitSetResize( MRBitSet* bs, size_t size, bool value );

/// ...
MRMESHC_API void mrBitSetAutoResizeSet( MRBitSet* bs, size_t pos, bool value );

/// ...
MRMESHC_API MRBitSet* mrBitSetSub( const MRBitSet* a, const MRBitSet* b );

/// deallocates a BitSet object
MRMESHC_API void mrBitSetFree( MRBitSet* bs );

/// creates a new FaceBitSet object
MRMESHC_API MRFaceBitSet* mrFaceBitSetNew( void );

/// creates a copy of a FaceBitSet object
MRMESHC_API MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs );

/// deallocates a FaceBitSet object
MRMESHC_API void mrFaceBitSetFree( MRFaceBitSet* fbs );

MR_EXTERN_C_END
