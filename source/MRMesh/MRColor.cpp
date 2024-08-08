#include "MRColor.h"


MR::Color MR::blend( const MR::Color& front, const MR::Color& back )
{
    const Vector4f frontColor4 = Vector4f( front );
    const Vector3f a = Vector3f( frontColor4.x, frontColor4.y, frontColor4.z ) * frontColor4.w;
    const Vector4f backColor4 = Vector4f( back );
    const Vector3f b = Vector3f( backColor4.x, backColor4.y, backColor4.z ) * backColor4.w * ( 1 - frontColor4.w );
    const float alphaRes = frontColor4.w + backColor4.w * ( 1 - frontColor4.w );
    const Vector3f newColor = ( a + b ) / alphaRes;
    return Color( newColor.x, newColor.y, newColor.z, alphaRes );
}
