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
#include "MRMeshComponents.h"
#include "MRLaplacian.h"
#include "MRLineSegm.h"

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
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
        newPoints = mesh.points;
        keepGoing = BitSetParallelFor( zone, [&] ( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            VertBitSet neighbors( mesh.topology.lastValidVert() + 1 );
            neighbors.set( v );

            dilateRegion( mesh, neighbors, surfaceRadius );

            PointAccumulator accum;
            int count = 0;
            for ( auto newV : neighbors )
            {
                Vector3d ptD = Vector3d( mesh.points[newV] );
                accum.addPoint( ptD );
                ++count;
            }
            if ( count < 6 )
                return;

            auto& np = newPoints[v];

            Vector3f target;
            if ( params.type == RelaxApproxType::Planar )
                target = accum.getBestPlanef().project( np );
            else if ( params.type == RelaxApproxType::Quadric )
            {
                AffineXf3d basis = accum.getBasicXf();
                basis.A = basis.A.transposed();
                std::swap( basis.A.x, basis.A.y );
                std::swap( basis.A.y, basis.A.z );
                basis.A = basis.A.transposed();
                auto basisInv = basis.inverse();

                QuadricApprox approxAccum;
                for ( auto newV : neighbors )
                    approxAccum.addPoint( basisInv( Vector3d( mesh.points[newV] ) ) );

                auto centerPoint = basisInv( Vector3d( mesh.points[v] ) );
                const auto coefs = approxAccum.calcBestCoefficients();
                centerPoint.z =
                    coefs[0] * centerPoint.x * centerPoint.x +
                    coefs[1] * centerPoint.x * centerPoint.y +
                    coefs[2] * centerPoint.y * centerPoint.y +
                    coefs[3] * centerPoint.x +
                    coefs[4] * centerPoint.y +
                    coefs[5];
                target = Vector3f( basis( centerPoint ) );
            }
            np += ( params.force * ( target - np ) );
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
        auto spikeVerts = mesh.findSpikeVertices( minSumAngle, region ).value();
        if ( spikeVerts.count() == 0 )
            break;
        relax( mesh, { { 1,&spikeVerts } } );
    }
}

void smoothRegionBoundary( Mesh & mesh, const FaceBitSet & regionFaces, int numIters )
{
    MR_TIMER
    assert( numIters > 0 );
    if ( !regionFaces.any() || numIters <= 0 )
        return;

    // value 1 for out-of-region vertices
    Vector<float,VertId> scalarField( mesh.topology.vertSize(), 1 );

    const auto regionVerts = getIncidentVerts( mesh.topology, regionFaces );
    // value -1 for in-region vertices
    for( auto v : regionVerts )
        scalarField[v] = -1;

    /// free vertices must have both in-region and out-of-region neighbor faces
    VertBitSet freeVerts = getIncidentVerts( mesh.topology, mesh.topology.getValidFaces() - regionFaces ) & regionVerts;

    for ( const auto & cc : MeshComponents::getAllComponentsVerts( mesh ) )
    {
        auto freeCC = cc & freeVerts;
        auto numfree = freeCC.count();
        auto numCC = cc.count();
        assert( numfree <= numCC );
        if ( numfree <= 0 )
            continue; // too small connected component
        if ( numfree < numCC )
            continue; // at least one fixed vertex in the component

        // all component vertices are free, just fix them all (to -1) to avoid under-determined system of equations
        freeVerts -= cc;
    }

    Laplacian lap( mesh );
    std::vector<Vector3f> newPos;
    for( int iter = 0; iter < numIters; ++iter )
    {
        lap.init( freeVerts, Laplacian::EdgeWeights::Cotan, Laplacian::RememberShape::No );
        lap.applyToScalar( scalarField );

        newPos.clear();
        newPos.reserve( freeVerts.count() );
        for ( const auto v : freeVerts )
        {
            const auto pt = mesh.points[v];
            Vector3f bestPos = pt;
            float bestDist2 = FLT_MAX;
            for ( auto e : orgRing( mesh.topology, v ) )
            {
                if ( !mesh.topology.left( e ) )
                    continue;

                VertId vs[3] = {
                    v,
                    mesh.topology.dest( e ),
                    mesh.topology.dest( mesh.topology.next( e ) )
                };

                for ( int i = 0; i < 3; ++i )
                {
                    const auto val  = scalarField[vs[0]];
                    const auto val1 = scalarField[vs[1]];
                    const auto val2 = scalarField[vs[2]];

                    if ( val * val1 >= 0 )
                        continue;
                    if ( val * val1 >= 0 )
                        continue;

                    LineSegm3f ls;
                    const float c1 = val / ( val - val1 );
                    ls.a = ( 1 - c1 ) * mesh.points[vs[0]] + c1 * mesh.points[vs[1]];
                    const float c2 = val / ( val - val2 );
                    ls.b = ( 1 - c2 ) * mesh.points[vs[0]] + c2 * mesh.points[vs[2]];

                    const auto proj = closestPointOnLineSegm( pt, ls );
                    const auto dist2 = ( pt - proj ).lengthSq();
                    if ( dist2 < bestDist2 )
                    {
                        bestPos = proj;
                        bestDist2 = dist2;
                    }

                    std::rotate( vs, vs + 1, vs + 3 );
                }
            }

            newPos.push_back( bestPos );
        }

        int n = 0;
        for ( const auto v : freeVerts )
            // 0.75 to reduce oscillation on the next iteration
            mesh.points[v] += 0.75f * ( newPos[n++] - mesh.points[v] );
    }
}

} //namespace MR
