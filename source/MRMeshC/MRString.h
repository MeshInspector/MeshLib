#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

const char* mrStringData( const MRString* str );

size_t mrStringSize( const MRString* str );

void mrStringFree( MRString* str );

#ifdef __cplusplus
}
#endif
