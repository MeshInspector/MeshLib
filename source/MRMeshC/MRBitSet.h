#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

MRMESHC_API void mrBitSetFree( MRBitSet* bs );

MR_EXTERN_C_END
