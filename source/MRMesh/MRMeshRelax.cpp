#include "MRMeshRelax.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MREdgePaths.h"
#include "MRBestFitQuadric.h"
#include "MRVector4.h"
#include "MRMeshFixer.h"
#include "MRRegionBoundary.h"
#include <MRPch/MROpenvdb.h>

namespace MR
{

bool relax( Mesh& mesh, const MeshRelaxParams& params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER;
    MR_WRITER( mesh );

    VertCoords newPoints;
    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    bool keepGoing = true;
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
        newPoints = mesh.points;
        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            Vector3d sum;
            int count = 0;
            for ( auto e : orgRing( mesh.topology, e0 ) )
            {
                sum += Vector3d( mesh.points[mesh.topology.dest( e )] );
                ++count;
            }
            auto& np = newPoints[v];
            auto pushForce = params.force * ( Vector3f{sum / double( count )} - np );
            np += pushForce;
        }, internalCb );
        mesh.points.swap( newPoints );
        if ( !keepGoing )
            break;
    }
    if ( keepGoing && params.hardSmoothTetrahedrons )
    {
        auto tetrahedrons = findNRingVerts( mesh.topology, 3, params.region );
        BitSetParallelFor( tetrahedrons, [&] ( VertId v )
        {
            Vector3f center;
            for ( auto e : orgRing( mesh.topology, v ) )
                center += mesh.destPnt( e );
            mesh.points[v] = center / 3.0f;
        } );
    }
    return keepGoing;
}

bool relaxKeepVolume( Mesh& mesh, const MeshRelaxParams& params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER;
    MR_WRITER( mesh );

    VertCoords newPoints;

    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    std::vector<Vector3f> vertPushForces( zone.size() );
    bool keepGoing = true;
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb1 = subprogress( cb, [&]( float p ) { return ( float( i ) + p * 0.5f ) / float( params.iterations ); } );
        auto internalCb2 = subprogress( cb, [&]( float p ) { return ( float( i ) + p * 0.5f + 0.5f ) / float( params.iterations ); } );
        newPoints = mesh.points;
        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            Vector3d sum;
            int count = 0;
            for ( auto e : orgRing( mesh.topology, e0 ) )
            {
                sum += Vector3d( mesh.points[mesh.topology.dest( e )] );
                ++count;
            }
            vertPushForces[v] = params.force * ( Vector3f{sum / double( count )} - mesh.points[v] );
        }, internalCb1 );
        if ( !keepGoing )
            break;
        keepGoing = BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;

            int count = 0;
            for ( [[maybe_unused]] auto e : orgRing( mesh.topology, e0 ) )
                ++count;

            auto& np = newPoints[v];
            np += vertPushForces[v];
            auto modifier = 1.0f / count;
            for ( auto e : orgRing( mesh.topology, e0 ) )
                np -= ( vertPushForces[mesh.topology.dest( e )] * modifier );
        }, internalCb2 );
        mesh.points.swap( newPoints );
        if ( !keepGoing )
            break;
    }
    if ( keepGoing && params.hardSmoothTetrahedrons )
    {
        auto tetrahedrons = findNRingVerts( mesh.topology, 3, params.region );
        BitSetParallelFor( tetrahedrons, [&] ( VertId v )
        {
            Vector3f center;
            for ( auto e : orgRing( mesh.topology, v ) )
                center += mesh.destPnt( e );
            mesh.points[v] = center / 3.0f;
        } );
    }
    return keepGoing;
}

bool relaxApprox( Mesh& mesh, const MeshApproxRelaxParams& params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;
    MR_TIMER;
    MR_WRITER( mesh );

    float surfaceRadius = ( params.surfaceDilateRadius <= 0.0f ) ?
        ( float( std::sqrt( mesh.area() ) ) * 1e-3f ) : params.surfaceDilateRadius;

    VertCoords newPoints;
    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    bool keepGoing = true;

    auto creaseEdges = mesh.findCreaseEdges( PI_F / 6 );
    auto creaseVerts = zone & getIncidentVerts( mesh.topology, creaseEdges );

    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
        newPoints = mesh.points;
        keepGoing = BitSetParallelFor( creaseVerts, [&] ( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            VertBitSet neighbors( mesh.topology.lastValidVert() + 1 );
            neighbors.set( v );

            dilateRegion( mesh, neighbors, surfaceRadius );

            using namespace openvdb;
            using namespace openvdb::tools;
            std::vector<Vec3d> points, normals;

            for ( auto newV : neighbors )
            {
                auto p = mesh.points[newV];
                points.emplace_back( p.x, p.y, p.z );
                auto n = mesh.normal( newV );
                normals.emplace_back( n.x, n.y, n.z );
            }
            if ( points.size() < 3 )
                return;

            auto f = findFeaturePoint( points, normals );

            auto& np = newPoints[v];
            np = Vector3f( (float)f.x(), (float)f.y(), (float)f.z() );
        }, internalCb );
        mesh.points.swap( newPoints );
        if ( !keepGoing )
            break;
    }
    if ( keepGoing && params.hardSmoothTetrahedrons )
    {
        auto tetrahedrons = findNRingVerts( mesh.topology, 3, params.region );
        BitSetParallelFor( tetrahedrons, [&] ( VertId v )
        {
            Vector3f center;
            for ( auto e : orgRing( mesh.topology, v ) )
                center += mesh.destPnt( e );
            mesh.points[v] = center / 3.0f;
        } );
    }
    return keepGoing;
}

void removeSpikes( Mesh & mesh, int maxIterations, float minSumAngle, const VertBitSet * region )
{
    if ( maxIterations <= 0 )
        return;

    MR_TIMER;

    for ( int i = 0; i < maxIterations; ++i )
    {
        auto spikeVerts = mesh.findSpikeVertices( minSumAngle, region );
        if ( spikeVerts.count() == 0 )
            break;
        relax( mesh, { { 1,&spikeVerts } } );
    }
}

} //namespace MR
