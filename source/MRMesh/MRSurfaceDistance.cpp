#include "MRSurfaceDistance.h"
#include "MRSurfaceDistanceBuilder.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

Vector<float,VertId> computeSurfaceDistances( const Mesh & mesh, const VertBitSet & startVertices, float maxDist, 
                                              const VertBitSet* region )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStartRegion( startVertices, 0 );
    while ( b.doneDistance() < maxDist )
    {
        b.growOne();
    }
    return b.takeDistanceMap();
}

Vector<float,VertId> computeSurfaceDistances( const Mesh & mesh, const VertBitSet & startVertices, const VertBitSet& targetVertices,
    float maxDist, const VertBitSet* region )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStartRegion( startVertices, 0 );

    auto toReachVerts = targetVertices - startVertices;
    auto toReachCount = toReachVerts.count();

    while ( toReachCount > 0 && b.doneDistance() < maxDist )
    {
        auto v = b.growOne();
        if ( toReachVerts.test( v ) )
            --toReachCount;
    }
    return b.takeDistanceMap();
}

Vector<float, VertId> computeSurfaceDistances( const Mesh& mesh, const HashMap<VertId, float>& startVertices, float maxDist,
                                               const VertBitSet* region )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStartVertices( startVertices );
    while ( b.doneDistance() < maxDist )
    {
        b.growOne();
    }
    return b.takeDistanceMap();
}

Vector<float,VertId> computeSurfaceDistances( const Mesh & mesh, const MeshTriPoint & start, const MeshTriPoint & end, 
                                              const VertBitSet* region, bool * endReached )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStart( start );

    VertId stopVerts[3];
    int numStopVerts = 0;
    if ( auto v = end.inVertex( mesh.topology ) )
    {
        stopVerts[0] = v;
        numStopVerts = 1;
    }
    else if ( auto e = end.onEdge( mesh.topology ) )
    {
        auto o = mesh.topology.org( e->e );
        auto d = mesh.topology.dest( e->e );
        stopVerts[0] = o;
        stopVerts[1] = d;
        numStopVerts = 2;
    }
    else
    {
        mesh.topology.getLeftTriVerts( end.e, stopVerts );
        numStopVerts = 3;
    }

    if ( endReached )
        *endReached = true;
    while ( numStopVerts > 0 )
    {
        auto v = b.growOne();
        if ( !v )
        {
            if ( endReached )
                *endReached = false;
            break;
        }

        auto it = std::find( stopVerts, stopVerts + numStopVerts, v );
        if ( it != stopVerts + numStopVerts )
        {
            for ( ; it + 1 < stopVerts + numStopVerts; ++it )
                *it = *(it + 1);
            --numStopVerts;
        }
    }
    return b.takeDistanceMap();
}

} //namespace MR
