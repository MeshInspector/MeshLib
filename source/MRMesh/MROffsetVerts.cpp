#include "MROffsetVerts.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRMeshFillHole.h"
#include "MRTimer.h"
#include "MRPositionVertsSmoothly.h"
#include "MRMapOrHashMap.h"
#include "MRBuffer.h"
#include "MRRingIterator.h"

namespace MR
{

bool offsetVerts( Mesh& mesh, const VertMetric& offset, const ProgressCallback& cb )
{
    MR_TIMER;
    mesh.invalidateCaches();

    // prepare all normals before modifying the points
    VertNormals ns( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        ns[v] = mesh.pseudonormal( v );
    } );

    return BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        mesh.points[v] += offset( v ) * ns[v];
    }, cb );
}

Mesh makeThickMesh( const Mesh & m, const ThickenParams & params )
{
    MR_TIMER;

    VertNormals dirs;
    dirs.resizeNoInit( m.topology.vertSize() );
    BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v )
    {
        dirs[v] = m.pseudonormal( v );
    } );

    const auto maxOffset = std::max( params.insideOffset, params.outsideOffset );
    if ( maxOffset > 0 )
    {
        Buffer<float, VertId> vertStabilizers( m.topology.vertSize() );
        Buffer<float, UndirectedEdgeId> edgeWeights( m.topology.undirectedEdgeSize() );

        BitSetParallelFor( m.topology.getValidVerts(), [&, rden = 1 / ( 2 * sqr( maxOffset ) )]( VertId v )
        {
            float vertStabilizer = 0;
            for ( auto e : orgRing( m.topology, v ) )
            {
                // gaussian, weight is 1 for very short edges (compared to offset) and 0 for very long edges
                auto edgeW = std::exp( -m.edgeLengthSq( e ) * rden );
                if ( e.even() ) //only one thread to write in undirected edge
                    edgeWeights[e] = edgeW;
                // stabilizer is 1 if all edges are long compared to offset, and 0 otherwise
                vertStabilizer = std::max( vertStabilizer, 1 - edgeW );
            }
            vertStabilizers[v] = vertStabilizer;
        } );

        /// smooth directions on original mesh to avoid boundary effects near stitches
        positionVertsSmoothlySharpBd( m.topology, dirs, PositionVertsSmoothlyParams
            {
                .vertStabilizers = [&vertStabilizers]( VertId v ) { return vertStabilizers[v]; },
                .edgeWeights = [&edgeWeights]( UndirectedEdgeId ue ) { return edgeWeights[ue]; }
            }
        );
        BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v )
        {
            dirs[v] = dirs[v].normalized();
        } );
    }

    Mesh res = m;
    auto holesRepr = m.topology.findHoleRepresentiveEdges();
    EdgeLoops mHoles( holesRepr.size() );
    EdgeLoops extHoles( holesRepr.size() );
    for ( int i = 0; i < holesRepr.size(); ++i )
    {
        mHoles[i] = trackRightBoundaryLoop( m.topology, holesRepr[i] );
        auto e = makeDegenerateBandAroundHole( res, holesRepr[i] );
        extHoles[i] = trackRightBoundaryLoop( res.topology, e );
    }
    PartMapping map;
    auto m2resVerts = VertMapOrHashMap::createMap();
    map.src2tgtVerts = &m2resVerts;
    res.addMeshPart( m, true, extHoles, mHoles, map );

    // apply shifts
    BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v )
    {
        res.points[v] += params.outsideOffset * dirs[v];
        auto resV = getAt( m2resVerts, v );
        if ( !resV )
        {
            assert( false );
            return;
        }
        res.points[resV] -= params.insideOffset * dirs[v];
    } );

    return res;
}

} //namespace MR
