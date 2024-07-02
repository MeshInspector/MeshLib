#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

MRMESHC_API size_t mrBitSetBlocksNum( const MRBitSet* bs );

MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

MRMESHC_API bool mrBitSetEq( const MRBitSet* a, const MRBitSet* b );

MRMESHC_API void mrBitSetFree( MRBitSet* bs );

MRMESHC_API MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs );

MRMESHC_API void mrFaceBitSetFree( MRFaceBitSet* fbs );

MR_EXTERN_C_END
