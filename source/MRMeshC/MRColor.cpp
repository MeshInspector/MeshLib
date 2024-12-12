#include "MRColor.h"
#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRColor.h"

using namespace MR;

REGISTER_AUTO_CAST( Color )

MR_VECTOR_LIKE_IMPL( VertColors, Color )

MRColor mrColorNew( void )
{
    RETURN( Color() );
}

MRColor mrColorFromComponents( uint8_t r, uint8_t g, uint8_t b, uint8_t a )
{
    RETURN( Color( r, g, b, a ) );
}

MRColor mrColorFromFloatComponents( float r, float g, float b, float a )
{
    RETURN( Color( r, g, b, a ) );
}

unsigned int mrColorGetUInt32( const MRColor* color_ )
{
    ARG( color );
    return color.getUInt32();
}

MRVertColors* mrVertColorsNewSized( size_t size )
{
    return reinterpret_cast< MRVertColors* >( new vector_wrapper<VertColors>( std::vector<VertColors>( size ) ) );
}