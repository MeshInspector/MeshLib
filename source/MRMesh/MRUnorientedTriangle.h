#pragma once

#include "MRId.h"
#include <cassert>
#include <utility>

namespace MR
{

/// describes a triangle as three vertex IDs sorted in a way to quickly find same triangles
/// irrespective of vertex order (clockwise or counterclockwise)
struct UnorientedTriangle : ThreeVertIds
{
    UnorientedTriangle( const ThreeVertIds & inVs,
        bool * outFlipped = nullptr ) ///< optional output: true if the orientation of the triangle has flipped
        : ThreeVertIds( inVs )
    {
        bool flipped = false;
        auto checkSwap = [this, &flipped]( int i, int j )
        {
            assert( i < j );
            assert( (*this)[i] != (*this)[j] );
            if ( (*this)[i] > (*this)[j] )
            {
                flipped = !flipped;
                std::swap( (*this)[i], (*this)[j] );
            }
        };
        checkSwap( 0, 1 );
        checkSwap( 0, 2 );
        checkSwap( 1, 2 );
        if ( outFlipped )
            *outFlipped = flipped;
    }

    /// returns this triangle with the opposite orientation
    ThreeVertIds getFlipped() const { return { (*this)[0], (*this)[2], (*this)[1] }; } // id #0 remains the lowest

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
            2 * size_t( triplet[0] ) +
            3 * size_t( triplet[1] ) +
            5 * size_t( triplet[2] );
    }
};

} //namespace std
