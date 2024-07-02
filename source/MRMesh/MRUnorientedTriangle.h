#pragma once

#include "MRId.h"
#include <cassert>
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
        return
            2 * size_t( triplet.verts[0] ) +
            3 * size_t( triplet.verts[1] ) +
            5 * size_t( triplet.verts[2] );
    }
};

} //namespace std
