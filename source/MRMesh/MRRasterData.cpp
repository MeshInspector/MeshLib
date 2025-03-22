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
    return visit( type, [] <typename T> ( T )
    {
        return sizeof(T);
    } );
}

size_t RasterData::sizeOf( SampleType type )
{
    switch ( type )
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
}

Expected<DistanceMap> RasterData::toDistanceMap( DistanceMapToWorld* dmapToWorld ) const
{
    if ( samples_ != SampleType::Scalar )
        return unexpected( "Cannot convert non-scalar raster to distance map" );

    DistanceMap result( dims_.x, dims_.y );
    assert( result.size() == size_ );
    visit( [&] <typename T> ( const T* data )
    {
        std::copy_n( data, size_, result.data() );
    } );

    if ( dmapToWorld )
        *dmapToWorld = xf();

    return result;
}

Expected<Image> RasterData::toImage() const
{
    if ( samples_ != SampleType::RGB && samples_ != SampleType::RGBA )
        return unexpected( "Cannot convert non-RGB raster to image" );

    Image result {
        .resolution = dims_,
    };
    result.pixels.resize( size_ );

    visit( [&] <typename T> ( const T* data )
    {
        switch ( samples_ )
        {
        case SampleType::RGBA:
            std::copy_n( data, size_ * 4, reinterpret_cast<uint8_t*>( result.pixels.data() ) );
            break;
        case SampleType::RGB:
            for ( auto i = 0u; i < size_; ++i )
            {
                result.pixels[i] = Color {
                    Color::valToUint8( data[i * 3 + 0] ),
                    Color::valToUint8( data[i * 3 + 1] ),
                    Color::valToUint8( data[i * 3 + 2] ),
                };
            }
            break;
        default:
            MR_UNREACHABLE_NO_RETURN
        }
    } );

    return result;
}

} // namespace MR
