#include "MRImageTransform.h"

namespace MR::ImageTransform
{

Image rotateClockwise90( const Image& image )
{
    Image newImage;
    newImage.resolution = { image.resolution.y, image.resolution.x };
    newImage.pixels.resize( newImage.resolution.x * newImage.resolution.y );
    for ( size_t j = 0; j < image.resolution.y; ++j )
        for ( size_t i = 0; i < image.resolution.x; ++i )
        {
            const size_t imageIndex = j * image.resolution.x + i;
            const size_t newImageIndex = ( image.resolution.x - 1 - i ) * image.resolution.y + j;
            newImage.pixels[newImageIndex] = image.pixels[imageIndex];
        }
    return newImage;
}

}