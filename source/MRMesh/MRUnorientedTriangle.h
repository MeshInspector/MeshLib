#pragma once

#include "MRId.h"
#include <cassert>
#include <cstdint>
#include <utility>

namespace MR
{

/// describes a triangle as three vertex IDs sorted in a way to quickly find same triangles
/// irrespective of vertex order (clockwise or counterclockwise)
struct UnorientedTriangle
{
    ThreeVertIds verts{};

    UnorientedTriangle( const ThreeVertIds & inVs,
        bool * outFlipped = nullptr ) ///< optional output: true if the orientation of the triangle has flipped
        : verts( inVs )
    {
        bool flipped = false;
        auto checkSwap = [this, &flipped]( int i, int j )
        {
            assert( i < j );
            assert( verts[i] != verts[j] );
            if ( verts[i] > verts[j] )
            {
                flipped = !flipped;
                std::swap( verts[i], verts[j] );
            }
        };
        checkSwap( 0, 1 );
        checkSwap( 0, 2 );
        checkSwap( 1, 2 );
        if ( outFlipped )
            *outFlipped = flipped;
    }

    /// returns this triangle with the opposite orientation
    ThreeVertIds getFlipped() const { return { verts[0], verts[2], verts[1] }; } // id #0 remains the lowest

    operator       ThreeVertIds &()       { return verts; }
    operator const ThreeVertIds &() const { return verts; }

          VertId &operator[]( std::size_t i )       { return verts[i]; }
    const VertId &operator[]( std::size_t i ) const { return verts[i]; }

    friend bool operator==( const UnorientedTriangle& a, const UnorientedTriangle& b ) = default;
};

} //namespace MR

namespace std
{

template <>
struct hash<MR::UnorientedTriangle>
{
    size_t operator() ( const MR::UnorientedTriangle& triplet ) const noexcept
    {
        // two different triangles get equal values only if some of their sorted vertex ids differ by 2^21 or more
        // (by 2^10 or more if size_t is 32-bit), unlike a sum of ids with small multipliers, which gives equal values to many triangles with close ids
        const auto a = std::uint32_t( triplet.verts[0] );
        const auto b = std::uint32_t( triplet.verts[1] );
        const auto c = std::uint32_t( triplet.verts[2] );
        if constexpr ( sizeof( size_t ) >= 8 )
        {
            // ids below 2^21 occupy non-overlapping bits
            return size_t( a ^ ( std::uint64_t( b ) << 21 ) ^ ( std::uint64_t( c ) << 42 ) );
        }
        else
        {
            // three ids do not fit in 32 bits, so they are combined with odd multipliers selected to maximize the difference above
            return size_t( 0xea87bb09u * a + 0xd8306b31u * b + c );
        }
    }
};

} //namespace std
