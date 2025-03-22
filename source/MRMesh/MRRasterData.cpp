#include "MRRasterData.h"

#include "MRAffineXf.h"
#include "MRMatrix3.h"

namespace MR
{

Vector3f RasterToWorld::toWorld( float x, float y, float value ) const
{
    return orgPoint_ + x * pixelXVec_ + y * pixelYVec_ + value * direction_;
}

AffineXf3f RasterToWorld::xf() const
{
    return { Matrix3f::fromColumns( pixelXVec_, pixelYVec_, direction_ ), orgPoint_ };
}

void RasterToWorld::setXf( const AffineXf3f& xf )
{
    orgPoint_ = xf.b;
    pixelXVec_ = xf.A.col( 0 );
    pixelYVec_ = xf.A.col( 1 );
    direction_ = xf.A.col( 2 );
}

size_t RasterData::sizeOf( DataType type )
{
    return visit( [] <typename T> ( T* )
    {
        return sizeof( T );
    }, type );
}

size_t RasterData::sizeOf( SampleType samples )
{
    switch ( samples )
    {
    case SampleType::Scalar:
        return 1;
    case SampleType::RGB:
        return 3;
    case SampleType::RGBA:
        return 4;
    }
    MR_UNREACHABLE
}

RasterData::RasterData( DataType type, SampleType samples )
    : type_( type )
    , samples_( samples )
{
    visit( [this] <typename T> ( T* )
    {
        variant_.emplace<T*>( (T*)nullptr );
    }, type_ );
}

RasterData::RasterData( const Vector2i& dims, DataType type, SampleType samples )
    : RectIndexer( dims )
    , type_( type )
    , samples_( samples )
{
    data_.resize( bytes() );
}

RasterData::RasterData( const Vector2i& dims, const AffineXf3f& xf, DataType type, SampleType samples )
    : RectIndexer( dims )
    , RasterToWorld( xf )
    , type_( type )
    , samples_( samples )
{
    data_.resize( bytes() );
}

void RasterData::resize( const Vector2i& dims )
{
    RectIndexer::resize( dims );
    data_.resize( bytes() );
    visit( [this] <typename T> ( T* )
    {
        variant_.emplace<T*>( reinterpret_cast<T*>( data_.data() ) );
    }, type_ );
}

DistanceMap RasterData::toDistanceMap( DistanceMapToWorld* dmapToWorld ) const
{
    DistanceMap result( dims_.x, dims_.y );

    std::visit( [&] <typename T> ( const T* data )
    {
        iterateSamples( samples_, data, size_, [&] ( size_t i, const T* sample )
        {
            auto& value = result.data()[i];
            switch ( samples_ )
            {
            case SampleType::Scalar:
                value = sample[0];
                break;
            case SampleType::RGB:
            case SampleType::RGBA:
                // luma/brightness component from the YCbCr color space
                value =
                      (float)sample[0] * 0.299f
                    + (float)sample[1] * 0.587f
                    + (float)sample[2] * 0.114f
                ;
                break;
            }
        } );
    }, variant_ );

    if ( dmapToWorld )
        *dmapToWorld = xf();

    return result;
}

Image RasterData::toImage() const
{
    Image result {
        .resolution = dims_,
    };
    result.pixels.resize( size_ );

    std::visit( [&] <typename T> ( const T* data )
    {
        iterateSamples( samples_, data, size_, [&] ( size_t i, const T* sample )
        {
            auto& pixel = result.pixels[i];
            switch ( samples_ )
            {
            case SampleType::Scalar:
                pixel = Color {
                    Color::valToUint8( sample[0] ),
                    Color::valToUint8( sample[0] ),
                    Color::valToUint8( sample[0] ),
                };
                break;
            case SampleType::RGB:
                pixel = Color {
                    Color::valToUint8( sample[0] ),
                    Color::valToUint8( sample[1] ),
                    Color::valToUint8( sample[2] ),
                };
                break;
            case SampleType::RGBA:
                pixel = Color {
                    Color::valToUint8( sample[0] ),
                    Color::valToUint8( sample[1] ),
                    Color::valToUint8( sample[2] ),
                    Color::valToUint8( sample[3] ),
                };
                break;
            }
        } );
    }, variant_ );

    return result;
}

} // namespace MR
