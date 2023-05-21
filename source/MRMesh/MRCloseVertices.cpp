#include "MRCloseVertices.h"
#include "MRMesh.h"
#include "MRAABBTreePoints.h"
#include "MRPointsInBall.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

VertMap findSmallestCloseVertices( const VertCoords & points, const VertBitSet & valid, float closeDist )
{
    MR_TIMER

    AABBTreePoints tree( points, valid );
    VertMap res;
    res.resizeNoInit( points.size() );
    BitSetParallelForAll( valid, [&]( VertId v )
    {
        VertId smallestCloseVert = v;
        if ( valid.test( v ) )
        {
            findPointsInBall( tree, points[v], closeDist, [&]( VertId cv, const Vector3f& )
            {
                if ( cv == v )
                    return;
                if ( res[cv] != cv )
                    return; // cv vertex is removed by itself
                smallestCloseVert = std::min( smallestCloseVert, cv );
            } );
        }
        res[v] = smallestCloseVert;
    } );

    return res;
}

VertMap findSmallestCloseVertices( const Mesh & mesh, float closeDist )
{
    return findSmallestCloseVertices( mesh.points, mesh.topology.getValidVerts(), closeDist );
}

} //namespace MR
