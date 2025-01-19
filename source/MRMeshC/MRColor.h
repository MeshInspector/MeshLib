#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// stores a color in 32-bit RGBA format
typedef struct MRColor
{
    uint8_t r, g, b, a;
} MRColor;
/// creates opaque black by default
MRMESHC_API MRColor mrColorNew(void);
/// creates color from byte components [0..255]
MRMESHC_API MRColor mrColorFromComponents( uint8_t r, uint8_t g, uint8_t b, uint8_t a );
/// creates color from float components [0..1]
MRMESHC_API MRColor mrColorFromFloatComponents( float r, float g, float b, float a );
/// returns color as unsigned int
MRMESHC_API unsigned int mrColorGetUInt32( const MRColor* color );

MR_VECTOR_LIKE_DECL( VertColors, Color )

/// needed for manual creation of a color map with given size
MRMESHC_API MRVertColors* mrVertColorsNewSized( size_t size );

MR_EXTERN_C_END
