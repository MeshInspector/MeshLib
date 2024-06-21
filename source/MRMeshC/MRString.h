#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

MRMESHC_API const char* mrStringData( const MRString* str );

MRMESHC_API size_t mrStringSize( const MRString* str );

MRMESHC_API void mrStringFree( MRString* str );

MR_EXTERN_C_END
