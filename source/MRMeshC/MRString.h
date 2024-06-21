#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

MRMESHC_API const char* mrStringData( const MRString* str );

MRMESHC_API size_t mrStringSize( const MRString* str );

MRMESHC_API void mrStringFree( MRString* str );

#ifdef __cplusplus
}
#endif
