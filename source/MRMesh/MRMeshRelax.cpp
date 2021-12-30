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

namespace MR
{

void relax( Mesh& mesh, const RelaxParams params )
{
    if ( params.iterations <= 0 )
        return;

    MR_TIMER;
    MR_MESH_WRITER( mesh );

    VertCoords newPoints;
    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    for ( int i = 0; i < params.iterations; ++i )
    {
        newPoints = mesh.points;
        BitSetParallelFor( zone, [&]( VertId v )
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
        } );
        mesh.points.swap( newPoints );
    }
}

void relaxKeepVolume( Mesh& mesh, const RelaxParams params )
{
    if ( params.iterations <= 0 )
        return;

    MR_TIMER;
    MR_MESH_WRITER( mesh );

    VertCoords newPoints;

    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    std::vector<Vector3f> vertPushForces( zone.size() );
    for ( int i = 0; i < params.iterations; ++i )
    {
        newPoints = mesh.points;
        BitSetParallelFor( zone, [&]( VertId v )
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
        } );
        BitSetParallelFor( zone, [&]( VertId v )
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
        } );
        mesh.points.swap( newPoints );
    }
}

void relaxApprox( Mesh& mesh, const MeshApproxRelaxParams params )
{
    if ( params.iterations <= 0 )
        return;
    MR_TIMER;
    MR_MESH_WRITER( mesh );

    float surfaceRadius = ( params.surfaceDilateRadius <= 0.0f ) ?
        ( float( std::sqrt( mesh.area() ) ) * 1e-3f ) : params.surfaceDilateRadius;

    VertCoords newPoints;
    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    for ( int i = 0; i < params.iterations; ++i )
    {
        newPoints = mesh.points;
        BitSetParallelFor( zone, [&] ( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            VertBitSet neighbors( mesh.topology.lastValidVert() + 1 );
            neighbors.set( v );

            dilateRegion( mesh, neighbors, surfaceRadius );

            PointAccumulator accum;
            Vector3d centroid;
            int count = 0;
            for ( auto newV : neighbors )
            {
                Vector3d ptD = Vector3d( mesh.points[newV] );
                centroid += ptD;
                accum.addPoint( ptD );
                ++count;
            }
            if ( count < 6 )
                return;

            auto& np = newPoints[v];
            centroid /= double( count );
            Vector3f target;
            auto plane = accum.getBestPlane();
            if ( params.type == RelaxApproxType::Planar )
            {
                target = Plane3f( plane ).project( np );
            }
            else if ( params.type == RelaxApproxType::Quadric )
            {
                AffineXf3d basis;
                basis.A.z = plane.n.normalized();
                auto [x, y] = basis.A.z.perpendicular();
                basis.A.x = x;
                basis.A.y = y;
                basis.A = basis.A.transposed();
                basis.b = Vector3d( np );
                auto basisInv = basis.inverse();
                QuadricApprox approxAccum;
                for ( auto newV : neighbors )
                    approxAccum.addPoint( basisInv( Vector3d( mesh.points[newV] ) ) );
                auto res = QuadricApprox::findZeroProjection( approxAccum.calcBestCoefficients() );
                target = Vector3f( basis( res ) );
            }
            np += ( params.force * ( 0.5f * target + Vector3f( 0.5 * centroid ) - np ) );
        } );
        mesh.points.swap( newPoints );
    }
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
        relax( mesh, { 1,&spikeVerts } );
    }
}

} //namespace MR
