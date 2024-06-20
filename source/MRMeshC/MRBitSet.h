#pragma once

#include "MRMeshFwd.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

size_t mrBitSetSize( const MRBitSet* bs );

void mrBitSetFree( MRBitSet* bs );

#ifdef __cplusplus
}
#endif
