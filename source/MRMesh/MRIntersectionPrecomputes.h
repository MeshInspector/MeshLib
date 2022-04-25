#pragma once
#include <boost/multiprecision/cpp_int.hpp>

#include "MRVector3.h"

namespace MR
{

// http://jcgt.org/published/0002/01/05/paper.pdf
// this functions finds index of maximum axis and stores it into dimZ
// dimX and dimY are filled by right-hand rule from dimZ
// Example input: dir = (1,1,-2). Result: dimZ = 2, dimX = 1, dimY = 0.
template <typename T>
void findMaxVectorDim( int& dimX, int& dimY, int& dimZ, const Vector3<T>& dir )
{
    if( dir.x > dir.y )
    {
        if( dir.x > dir.z )
        {
            if( dir.y > dir.z )
            {
                // x>y>z
                if( -dir.z > dir.x )
                {
                    dimZ = 2; dimX = 1; dimY = 0;
                }
                else
                {
                    dimZ = 0; dimX = 1; dimY = 2;
                }
            }
            else
            {
                // x>z>y
                if( -dir.y > dir.x )
                {
                    dimZ = 1; dimX = 0; dimY = 2;
                }
                else
                {
                    dimZ = 0; dimX = 1; dimY = 2;
                }
            }
        }
        else
        {
            // z>x>y
            if( -dir.y > dir.z )
            {
                dimZ = 1; dimX = 0; dimY = 2;
            }
            else
            {
                dimZ = 2; dimX = 0; dimY = 1;
            }
        }
    }
    else
    {
        if( dir.y > dir.z )
        {
            if( dir.x < dir.z )
            {
                // y>z>x
                if( -dir.x > dir.y )
                {
                    dimZ = 0; dimX = 2; dimY = 1;
                }
                else
                {
                    dimZ = 1; dimX = 2; dimY = 0;
                }
            }
            else
            {
                // y>x>z
                if( -dir.z > dir.y )
                {
                    dimZ = 2; dimX = 1; dimY = 0;
                }
                else
                {
                    dimZ = 1; dimX = 2; dimY = 0;
                }
            }
        }
        else
        {
            // z>y>x
            if( -dir.x > dir.z )
            {
                dimZ = 0; dimX = 2; dimY = 1;
            }
            else
            {
                dimZ = 2; dimX = 0; dimY = 1;
            }
        }
    }
}

// stores useful precomputed values for presented direction vector
// allows to avoid repeatable computations during intersection finding
template<typename T>
struct IntersectionPrecomputes
{
    // {1.f / dir}
    Vector3<T> invDir;
    // [0]max, [1]next, [2]next-next
    // f.e. {1,2,-3} => {2,1,0}
    int maxDimIdxZ = 2;
    int idxX = 0;
    int idxY = 1;

    // stores signs of direction vector;
    Vector3i sign;

    // precomputed factors
    T Sx, Sy, Sz;
    IntersectionPrecomputes() = default;
    IntersectionPrecomputes( const Vector3<T>& dir )
    {
        findMaxVectorDim( idxX, idxY, maxDimIdxZ, dir );

        sign.x = dir.x >= T( 0 ) ? 1 : 0;
        sign.y = dir.y >= T( 0 ) ? 1 : 0;
        sign.z = dir.z >= T( 0 ) ? 1 : 0;

        Sx = dir[idxX] / dir[maxDimIdxZ];
        Sy = dir[idxY] / dir[maxDimIdxZ];
        Sz = T( 1 ) / dir[maxDimIdxZ];

        invDir.x = ( dir.x == 0 ) ? std::numeric_limits<T>::max() : T( 1 ) / dir.x;
        invDir.y = ( dir.y == 0 ) ? std::numeric_limits<T>::max() : T( 1 ) / dir.y;
        invDir.z = ( dir.z == 0 ) ? std::numeric_limits<T>::max() : T( 1 ) / dir.z;
    }

};

/* CPU(X86_64) - AMD64 / Intel64 / x86_64 64-bit */
#if defined(__x86_64__) || defined(_M_X64)
template<>
struct IntersectionPrecomputes<float>
{
    // {1.f / dir}
    __m128 invDir;
    // [0]max, [1]next, [2]next-next
    // f.e. {1,2,-3} => {2,1,0}
    int maxDimIdxZ = 2;
    int idxX = 0;
    int idxY = 1;

    // precomputed factors
    float Sx, Sy, Sz;
    IntersectionPrecomputes() = default;
    IntersectionPrecomputes( const Vector3<float>& dir )
    {
        findMaxVectorDim( idxX, idxY, maxDimIdxZ, dir );

        Sx = dir[idxX] / dir[maxDimIdxZ];
        Sy = dir[idxY] / dir[maxDimIdxZ];
        Sz = float( 1 ) / dir[maxDimIdxZ];

        invDir = _mm_set_ps( 
            ( dir.x == 0 ) ? std::numeric_limits<float>::max() : 1 / dir.x, 
            ( dir.y == 0 ) ? std::numeric_limits<float>::max() : 1 / dir.y, 
            ( dir.z == 0 ) ? std::numeric_limits<float>::max() : 1 / dir.z, 
            1 );
    }

};
#else
    #pragma message("IntersectionPrecomputes<float>: no hardware optimized instructions")
#endif
}
