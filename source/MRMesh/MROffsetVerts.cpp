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
#include "MRMeshProject.h"
#include "MRBall.h"

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
            float vertStabilizer = 1;
            for ( auto e : orgRing( m.topology, v ) )
            {
                // gaussian, weight is 1 for very short edges (compared to offset) and 0 for very long edges
                auto edgeW = std::exp( -m.edgeLengthSq( e ) * rden );
                if ( e.even() ) //only one thread to write in undirected edge
                    edgeWeights[e] = edgeW;
                // stabilizer is 1 if all edges are long compared to offset, and 0 otherwise
                vertStabilizer = std::min( vertStabilizer, 1 - edgeW );
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

std::optional<VertScalars> findZcompensationShifts( const Mesh& mesh, const ZCompensateParams& params )
{
    MR_TIMER;
    assert( !params.reduceSelfIntersections || params.minThickness >= 0 );

    VertScalars zShifts( mesh.topology.vertSize() );
    (void)mesh.getAABBTree(); //prepare for findTrisInBall
    if ( !BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        const auto n = mesh.pseudonormal( v );
        if ( n.z >= 0 )
            return;

        auto vShift = params.maxShift * -n.z;
        if ( params.reduceSelfIntersections )
        {
            auto notIncidentFaces = [&mesh, v]( FaceId f )
            {
                VertId a, b, c;
                mesh.topology.getTriVerts( f, a, b, c );
                return v != a && v != b && v != c;
            };

            findTrisInBall( mesh, Ball3f{ mesh.points[v], sqr( vShift ) },
                [&]( const MeshProjectionResult & found, Ball3f & ball )
                {
                    if ( found.proj.point.z < mesh.points[v].z )
                        return Processing::Continue; // ignore triangles below point[v]
                    float dist = std::sqrt( found.distSq );
                    float newShift = std::max( 0.0f, dist - params.minThickness );
                    assert( newShift <= vShift );
                    vShift = newShift;
                    ball = Ball3f{ mesh.points[v], sqr( vShift ) };
                    return Processing::Continue;
                }, notIncidentFaces );

            // assuming that only this vertex shifts, verify that no incident triangle flips its normal
            constexpr int maxTries = 10;
            for ( int it = 0; it < maxTries; ++it )
            {
                bool flipDetected = false;
                for ( EdgeId e : orgRing( mesh.topology, v ) )
                {
                    if ( !mesh.topology.left( e ) )
                        continue;
                    auto ps = mesh.getLeftTriPoints( e );
                    const auto c0 = cross( ps[1] - ps[0], ps[2] - ps[0] );
                    ps[0].z += vShift;
                    const auto c1 = cross( ps[1] - ps[0], ps[2] - ps[0] );
                    if ( dot( c0, c1 ) < 0 )
                    {
                        flipDetected = true;
                        break;
                    }
                }
                if ( !flipDetected )
                    break;
                // flip detected, reduce shift amount twofold
                vShift *= 0.5;
            }
        }
        zShifts[v] = vShift;
    }, params.progress ) )
        return {};

    return zShifts;
}

MRMESH_API std::optional<VertCoords> findZcompensatedPositions( const Mesh& mesh, const ZCompensateParams& params0 )
{
    MR_TIMER;

    auto params = params0;
    params.progress = subprogress( params0.progress, 0.0f, 0.5f );
    auto maybeZShifts = findZcompensationShifts( mesh, params );
    if ( !maybeZShifts )
        return {};
    const auto& zShifts = *maybeZShifts;

    VertCoords res;
    res.resizeNoInit( mesh.points.size() );
    if ( !BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        auto p = mesh.points[v];
        p.z += zShifts[v];
        res[v] = p;
    }, subprogress( params0.progress, 0.5f, 1.0f ) ) )
        return {};
    return res;
}

bool zCompensate( Mesh& mesh, const ZCompensateParams& params0 )
{
    MR_TIMER;

    // prepare all shifts before modifying the points
    auto params = params0;
    params.progress = subprogress( params0.progress, 0.0f, 0.5f );
    auto maybeZShifts = findZcompensationShifts( mesh, params );
    if ( !maybeZShifts )
        return false;
    const auto& zShifts = *maybeZShifts;

    mesh.invalidateCaches();
    return BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        mesh.points[v].z += zShifts[v];
    }, subprogress( params0.progress, 0.5f, 1.0f ) );
}

} //namespace MR
