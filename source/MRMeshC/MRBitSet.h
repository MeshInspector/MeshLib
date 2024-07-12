#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// gets read-only access to the underlying blocks of a bitset
MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

/// gets count of the underlying blocks of a bitset
MRMESHC_API size_t mrBitSetBlocksNum( const MRBitSet* bs );

/// gets total length of a bitset
MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

/// checks if two bitsets are equal (have the same length and identical bit values)
MRMESHC_API bool mrBitSetEq( const MRBitSet* a, const MRBitSet* b );

/// deallocates a BitSet object
MRMESHC_API void mrBitSetFree( MRBitSet* bs );

/// creates a copy of a FaceBitSet object
MRMESHC_API MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs );

/// deallocates a FaceBitSet object
MRMESHC_API void mrFaceBitSetFree( MRFaceBitSet* fbs );

MR_EXTERN_C_END
