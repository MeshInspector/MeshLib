#include "MRRenderHelpers.h"

namespace MR
{

Vector2i calcTextureRes( int bufferSize, int maxTextWidth )
{
    if ( bufferSize <= maxTextWidth )
        return { bufferSize, 1 };
    int space = maxTextWidth - ( bufferSize % maxTextWidth );
    int height = ( bufferSize + maxTextWidth - 1 ) / maxTextWidth;
    if ( space == maxTextWidth )
        return { maxTextWidth, height };
    int shift = space / height;
    return { maxTextWidth - shift, height };
}

}
