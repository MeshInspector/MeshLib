#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// gets read-only access to the string data
MRMESHC_API const char* mrStringData( const MRString* str );

/// gets total length of the string
MRMESHC_API size_t mrStringSize( const MRString* str );

/// deallocates the string object
MRMESHC_API void mrStringFree( MRString* str );

MR_EXTERN_C_END
