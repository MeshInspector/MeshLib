#pragma once

#include "MRId.h"
#include <algorithm>

namespace MR
{

/// describes a triangle as three vertex IDs sorted in a way to quickly find same triangles
/// irrespective of vertex order (clockwise or counterclockwise)
struct UnorientedTriangle : ThreeVertIds
{
    UnorientedTriangle( const ThreeVertIds & inVs ) : ThreeVertIds( inVs )
    {
        std::sort( begin(), end() );
    }
    friend bool operator==( const UnorientedTriangle& a, const UnorientedTriangle& b ) = default;
};

/// defines hash function for UnorientedTriangle
struct UnorientedTriangleHasher
{
    size_t operator()( const UnorientedTriangle& triplet ) const
    {
        return 
            2 * size_t( triplet[0] ) +
            3 * size_t( triplet[1] ) +
            5 * size_t( triplet[2] );
    }
};

} //namespace MR
