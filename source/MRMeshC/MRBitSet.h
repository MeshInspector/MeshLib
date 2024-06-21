#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
#include <cstdint>

extern "C"
{
#else
#include <stdint.h>
#endif

MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

MRMESHC_API void mrBitSetFree( MRBitSet* bs );

#ifdef __cplusplus
}
#endif
