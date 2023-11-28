#pragma once

#include "MRColor.h"
#include "MRVector2.h"
#include "MRHeapBytes.h"
#include <cassert>
#include <vector>

namespace MR
{

/// struct to hold Image data
/// \ingroup BasicStructuresGroup
struct Image
{
    std::vector<Color> pixels;
    Vector2i resolution;

    /// fetches some texture element specified by integer coordinates
    [[nodiscard]] Color& operator[]( const Vector2i & p );
    [[nodiscard]] Color operator[]( const Vector2i & p ) const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return MR::heapBytes( pixels ); }

    /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
    /// returns the color of the closest pixel
    [[nodiscard]] MRMESH_API Color sampleDiscrete( const UVCoord & pos ) const;

    /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
    /// returns bilinear interpolated color at it
    [[nodiscard]] MRMESH_API Color sampleBilinear( const UVCoord & pos ) const;

    /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
    /// returns sampled color at it according to given filter
    [[nodiscard]] Color sample( FilterType filter, const UVCoord & pos ) const
        { return filter == FilterType::Discrete ? sampleDiscrete( pos ) : sampleBilinear( pos ); }
};

inline Color& Image::operator[]( const Vector2i & p )
{
    assert( p.x >= 0 && p.x < resolution.x );
    assert( p.y >= 0 && p.y < resolution.y );
    return pixels[p.x + p.y * resolution.x];
}

inline Color Image::operator[]( const Vector2i & p ) const
{
    assert( p.x >= 0 && p.x < resolution.x );
    assert( p.y >= 0 && p.y < resolution.y );
    return pixels[p.x + p.y * resolution.x];
}

} //namespace MR
