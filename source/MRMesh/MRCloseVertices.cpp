#include "MRCloseVertices.h"
#include "MRMesh.h"
#include "MRAABBTreePoints.h"
#include "MRPointsInBall.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

VertMap findSmallestCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid )
{
    MR_TIMER

    AABBTreePoints tree( points, valid );
    VertMap res;
    res.resizeNoInit( points.size() );
    ParallelFor( points, [&]( VertId v )
    {
        VertId smallestCloseVert = v;
        if ( !valid || valid->test( v ) )
        {
            findPointsInBall( tree, points[v], closeDist, [&]( VertId cv, const Vector3f& )
            {
                if ( cv == v )
                    return;
                smallestCloseVert = std::min( smallestCloseVert, cv );
            } );
        }
        res[v] = smallestCloseVert;
    } );
    // after parallel pass, some close vertices can be mapped further

    for ( auto v = 0_v; v < points.size(); ++v )
    {
        if ( valid && !valid->test( v ) )
            continue;
        VertId smallestCloseVert = res[v];
        if ( smallestCloseVert == v )
            continue; // v is the smallest closest by itself
        if ( res[smallestCloseVert] == smallestCloseVert )
            continue; // smallestCloseVert is not mapped further

        // find another closest
        smallestCloseVert = v;
        findPointsInBall( tree, points[v], closeDist, [&]( VertId cv, const Vector3f& )
        {
            if ( cv == v )
                return;
            if ( res[cv] != cv )
                return; // cv vertex is removed by itself
            smallestCloseVert = std::min( smallestCloseVert, cv );
        } );
        res[v] = smallestCloseVert;
    }

    return res;
}

VertMap findSmallestCloseVertices( const Mesh & mesh, float closeDist )
{
    return findSmallestCloseVertices( mesh.points, closeDist, &mesh.topology.getValidVerts() );
}

VertBitSet findCloseVertices( const VertCoords & points, float closeDist, const VertBitSet * valid )
{
    MR_TIMER
    VertBitSet res;
    const auto map = findSmallestCloseVertices( points, closeDist, valid );
    for ( auto v = 0_v; v < points.size(); ++v )
    {
        if ( const auto m = map[v]; m != v )
        {
            res.autoResizeSet( v );
            assert( m < v );
            res.autoResizeSet( m );
        }
    }
    return res;
}

VertBitSet findCloseVertices( const Mesh & mesh, float closeDist )
{
    return findCloseVertices( mesh.points, closeDist, &mesh.topology.getValidVerts() );
}

} //namespace MR
