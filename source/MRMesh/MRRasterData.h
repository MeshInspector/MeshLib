#pragma once

#include "MRMeshFwd.h"
#include "MRBuffer.h"
#include "MRDistanceMap.h"
#include "MRExpected.h"
#include "MRRectIndexer.h"
#include "MRVector3.h"

namespace MR
{

/// ...
template <typename... Args>
constexpr auto makeVariantArray( std::variant<Args...> )
{
    return std::array<std::variant<Args...>, sizeof...( Args )> { Args{}... };
}
template <typename Variant>
constexpr auto makeVariantArray()
{
    return makeVariantArray( Variant{} );
}

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
    using DataTypeVariant = std::variant<
        uint8_t*,
        uint16_t*,
        uint32_t*,
        uint64_t*,
        int8_t*,
        int16_t*,
        int32_t*,
        int64_t*,
        float*,
        double*
    >;
    /// ...
    template <typename T>
    static DataType from()
    {
        static constexpr auto variant = DataTypeVariant( (T*)nullptr );
        return DataType( variant.index() );
    }
    /// ...
    template <typename Visitor>
    static auto visit( DataType type, Visitor&& vis )
    {
        static constexpr auto variants = makeVariantArray<DataTypeVariant>();
        return std::visit( vis, variants.at( (size_t)type ) );
    }
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

    /// ...
    [[nodiscard]] MRMESH_API Expected<DistanceMap> toDistanceMap( DistanceMapToWorld* dmapToWorld = nullptr ) const;

    /// ...
    [[nodiscard]] MRMESH_API Expected<Image> toImage() const;

private:
    Buffer<std::byte> data_;
    DataTypeVariant variant_;
    DataType type_;
    SampleType samples_{ SampleType::Scalar };
};

} // namespace MR
