#pragma once

#include "MRMeshFwd.h"
#include "MRBuffer.h"
#include "MRDistanceMap.h"
#include "MRExpected.h"
#include "MRRectIndexer.h"
#include "MRVector3.h"

namespace MR
{

/// This structure store data to transform raster to world coordinates
class RasterToWorld
{
public:
    RasterToWorld() = default;
    explicit RasterToWorld( const AffineXf3f& xf ) { setXf( xf ); }

    /// get world coordinate by raster info
    /// x - float X coordinate of raster: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)
    /// y - float Y coordinate of raster: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)
    /// value - float value (value in raster)
    [[nodiscard]] MRMESH_API Vector3f toWorld( float x, float y, float value = 0.f ) const;
    [[nodiscard]] Vector3f toWorld( const Vector3f& p ) const { return toWorld( p.x, p.y, p.z ); }
    [[nodiscard]] Vector3f toWorld( const Vector2f& p ) const { return toWorld( p.x, p.y, 0.f ); }

    /// converts in transformation X: X(p) == toWorld( p.x, p.y, p.z )
    [[nodiscard]] MRMESH_API AffineXf3f xf() const;

    /// ...
    MRMESH_API void setXf( const AffineXf3f& xf );

private:
    /// world coordinates of distance map origin corner
    Vector3f orgPoint_;
    /// vector in world space of pixel x positive direction
    /// length is equal to pixel size
    /// \note typically it should be orthogonal to `pixelYVec`
    Vector3f pixelXVec_{ Vector3f::plusX() };
    /// vector in world space of pixel y positive direction
    /// length is equal to pixel size
    /// \note typically it should be orthogonal to `pixelXVec`
    Vector3f pixelYVec_{ Vector3f::plusY() };
    /// vector of depth direction
    /// \note typically it should be normalized and orthogonal to `pixelXVec` `pixelYVec` plane
    Vector3f direction_{ Vector3f::plusZ() };
};

class RasterData : public RectIndexer, public RasterToWorld
{
public:
    /// ...
    enum class DataType
    {
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float32,
        Float64,
    };
    /// ...
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    static DataType from();
    /// ...
    template <typename Visitor>
    static auto visit( DataType type, Visitor&& visitor );
    /// ...
    MRMESH_API static size_t sizeOf( DataType type );

    /// ...
    enum class SampleType
    {
        /// ...
        Scalar,
        /// ...
        RGB,
        /// ...
        RGBA,
    };
    /// ...
    static size_t sizeOf( SampleType type );

    MRMESH_API explicit RasterData( DataType type = DataType::UInt8, SampleType samples = SampleType::Scalar );
    MRMESH_API explicit RasterData( const Vector2i& dims, DataType type = DataType::UInt8, SampleType samples = SampleType::Scalar );
    MRMESH_API explicit RasterData( const Vector2i& dims, const AffineXf3f& xf, DataType type = DataType::UInt8, SampleType samples = SampleType::Scalar );

    /// ...
    [[nodiscard]] const std::byte* data() const { return data_.data(); }
    [[nodiscard]] std::byte* data() { return data_.data(); }
    /// ...
    [[nodiscard]] size_t bytes() const { return size_ * sizeOf( type_ ) * sizeOf( samples_ ); }
    /// ...
    [[nodiscard]] DataType type() const { return type_; }
    /// ...
    [[nodiscard]] SampleType samples() const { return samples_; }

    /// ...
    void resize( const Vector2i& dims );

    template <typename Visitor>
    auto visit( Visitor&& visitor ) const
    {
        return visit( type_, [&] <typename T> ( T )
        {
            return visitor( reinterpret_cast<const T*>( data_.data() ) );
        } );
    }
    template <typename Visitor>
    auto visit( Visitor&& visitor )
    {
        return visit( type_, [&] <typename T> ( T )
        {
            return visitor( reinterpret_cast<T*>( data_.data() ) );
        } );
    }

    /// ...
    [[nodiscard]] MRMESH_API Expected<DistanceMap> toDistanceMap( DistanceMapToWorld* dmapToWorld = nullptr ) const;

    /// ...
    [[nodiscard]] MRMESH_API Expected<Image> toImage() const;

private:
    Buffer<std::byte> data_;
    DataType type_{ DataType::UInt8 };
    SampleType samples_{ SampleType::Scalar };
};

template <typename T, typename>
RasterData::DataType RasterData::from()
{
#define RETURN_IF( TypeName, enumValue ) if constexpr ( std::is_same_v<T, TypeName> ) return DataType::enumValue;
    RETURN_IF( uint8_t, UInt8 )
    RETURN_IF( uint16_t, UInt16 )
    RETURN_IF( uint32_t, UInt32 )
    RETURN_IF( uint64_t, UInt64 )
    RETURN_IF( int8_t, Int8 )
    RETURN_IF( int16_t, Int16 )
    RETURN_IF( int32_t, Int32 )
    RETURN_IF( int64_t, Int64 )
    RETURN_IF( float, Float32 )
    RETURN_IF( double, Float64 )
#undef RETURN_IF
    MR_UNREACHABLE
}

template <typename Visitor>
auto RasterData::visit( DataType type, Visitor&& visitor )
{
#define VISIT_IF( enumValue, TypeName ) case DataType::enumValue: return visitor( TypeName() );
    switch ( type )
    {
        VISIT_IF( UInt8, uint8_t )
        VISIT_IF( UInt16, uint16_t )
        VISIT_IF( UInt32, uint32_t )
        VISIT_IF( UInt64, uint64_t )
        VISIT_IF( Int8, int8_t )
        VISIT_IF( Int16, int16_t )
        VISIT_IF( Int32, int32_t )
        VISIT_IF( Int64, int64_t )
        VISIT_IF( Float32, float )
        VISIT_IF( Float64, double )
    }
    MR_UNREACHABLE_NO_RETURN
#undef VISIT_IF
}

} // namespace MR
