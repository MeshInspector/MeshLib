#pragma once

#include "MRVector2.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/**
 * \brief finds index of maximum axis and stores it into dimY
 * \details http://jcgt.org/published/0002/01/05/paper.pdf
 * Example input: dir = (1,-2). Result: dimY = 1, dimX = 0.
 * \param[out] dimX index of minimum axis
 * \param[out] dimY index of maximum axis
 */
template <typename T>
void findMaxVectorDim( int& dimX, int& dimY, const Vector2<T>& dir )
{
    if( std::abs( dir.x ) >= std::abs( dir.y ) )
    {
        dimX = 1; dimY = 0;
    }
    else
    {
        dimX = 0; dimY = 1;
    }
}

/// stores useful precomputed values for presented direction vector
/// \details allows to avoid repeatable computations during intersection finding
template<typename T>
struct IntersectionPrecomputes2
{
    // {1 / dir}
    Vector2<T> invDir;
    // [0]max, [1]next, [2]next-next
    // f.e. {1,2} => {1,0}
    int maxDimIdxY = 1;
    int idxX = 0;

    /// stores signs of direction vector;
    Vector2i sign;

    /// precomputed factors
    T Sx, Sy;
    IntersectionPrecomputes2() = default;
    IntersectionPrecomputes2( const Vector2<T>& dir )
    {
        findMaxVectorDim( idxX, maxDimIdxY, dir );

        sign.x = dir.x >= T( 0 ) ? 1 : 0;
        sign.y = dir.y >= T( 0 ) ? 1 : 0;

        Sx = dir[idxX] / dir[maxDimIdxY];
        Sy = T( 1 ) / dir[maxDimIdxY];

        invDir.x = ( dir.x == 0 ) ? std::numeric_limits<T>::max() : T( 1 ) / dir.x;
        invDir.y = ( dir.y == 0 ) ? std::numeric_limits<T>::max() : T( 1 ) / dir.y;
    }
};

/// \}

}
