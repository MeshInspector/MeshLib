#include "MRMeshTotalAngle.h"
#include "MRMeshDelone.h"
#include "MRMesh.h"
#include "MRTriMath.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

namespace
{

// given quadrangle ABCD, computes AC * |angle(AC)|
float calcAngleLength( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d )
{
    const auto ln = dirDblArea( a, c, d );
    const auto rn = dirDblArea( a, b, c );
    const auto ac = c - a;
    return ac.length() * std::abs( MR::dihedralAngle( ln, rn, ac ) );
}

// returns not-negative value:
// 0 means huge improvement in mixed total angle and Delaunay criterion after flip,
// 1 means that that the flip will result in exactly same value of the criterion,
// FLT_MAX means huge degradation of the mixed criterion after flip
float totalAngleIncreaseOnFlip( const MeshTopology & topology, const VertCoords & points, EdgeId e, const ReduceTotalAngleParams & params )
{
    const auto can = canFlipEdge( topology, e, params.region, params.notFlippable, params.vertRegion );
    if ( can == FlipEdge::Cannot )
        return FLT_MAX;
    if ( can == FlipEdge::Must )
        return 0;

    //          g
    //       /    \
    //     d < e2 - c
    //   / ^      ^ | \
    //  h  e3   e  e1  f
    //   \ |  /     v /
    //     a - e0 > b
    //       \    /
    //         ep

    VertId av, bv, cv, dv;
    topology.getLeftTriVerts( e, av, cv, dv );
    const auto e0 = topology.prev( e );
    bv = topology.dest( e0 );

    const auto a = points[av];
    const auto b = points[bv];
    const auto c = points[cv];
    const auto d = points[dv];
    
    float oldAngleLength = calcAngleLength( a, b, c, d );
    float newAngleLength = calcAngleLength( b, c, d, a );
    if ( topology.right( e0 ) )
    {
        auto ep = points[topology.dest( topology.prev( e0 ) )];
        newAngleLength += calcAngleLength( a, ep, b, d );
        oldAngleLength += calcAngleLength( a, ep, b, c );
    }
    if ( auto e1 = topology.next( e.sym() ); topology.left( e1 ) )
    {
        auto f = points[topology.dest( topology.next( e1 ) )];
        newAngleLength += calcAngleLength( b, f, c, d );
        oldAngleLength += calcAngleLength( b, f, c, a );
    }
    if ( auto e2 = topology.prev( e.sym() ); topology.right( e2 ) )
    {
        auto g = points[topology.dest( topology.prev( e2 ) )];
        newAngleLength += calcAngleLength( c, g, d, b );
        oldAngleLength += calcAngleLength( c, g, d, a );
    }
    if ( auto e3 = topology.next( e ); topology.left( e3 ) )
    {
        auto h = points[topology.dest( topology.next( e3 ) )];
        newAngleLength += calcAngleLength( d, h, a, b );
        oldAngleLength += calcAngleLength( d, h, a, c );
    }

    const auto oldCircumDiameter = std::sqrt( std::max( circumcircleDiameterSq( a, c, d ), circumcircleDiameterSq( c, a, b ) ) );
    const auto newCircumDiameter = std::sqrt( std::max( circumcircleDiameterSq( b, d, a ), circumcircleDiameterSq( d, b, c ) ) );

    // ( 1 - f ) * newAngleLength / oldAngleLength + f * newCircumDiameter / oldCircumDiameter
    const auto f = params.factorDelone;
    float res;
    if ( oldAngleLength == 0 )
    {
        if ( newAngleLength == 0 )
            res = 1 - f;
        else
            return FLT_MAX;
    }
    else
        res = ( 1 - f ) * newAngleLength / oldAngleLength;

    if ( oldCircumDiameter == 0 )
    {
        if ( newCircumDiameter == 0 )
            res += f;
        else
            return FLT_MAX;
    }
    else
        res += f * newCircumDiameter / oldCircumDiameter;

    return res;
}

} //anonymous namespace

int reduceTotalAngle( MeshTopology& topology, const VertCoords& points, int numIters, const ReduceTotalAngleParams& region, const ProgressCallback& progressCallback )
{
    if ( numIters <= 0 )
        return 0;
    MR_TIMER;

    UndirectedEdgeBitSet flipCandidates( topology.undirectedEdgeSize() );
    UndirectedEdgeBitSet nextFlipCandidates( topology.undirectedEdgeSize(), true );

    int flipsDone = 0;
    for ( int iter = 0; iter < numIters; ++iter )
    {
        if ( progressCallback && !progressCallback( float( iter ) / numIters ) )
            return flipsDone;

        flipCandidates.reset();
        BitSetParallelFor( nextFlipCandidates, [&] ( UndirectedEdgeId e )
        {
            if ( totalAngleIncreaseOnFlip( topology, points, e, region ) < 1 )
                flipCandidates.set( e );
        } );
        nextFlipCandidates.reset();
        int flipsDoneBeforeThisIter = flipsDone;
        for ( UndirectedEdgeId ue : flipCandidates )
        {
            const EdgeId e = ue;
            if ( totalAngleIncreaseOnFlip( topology, points, e, region ) >= 1 )
                continue;

            ++flipsDone;
            topology.flipEdge( e );

            if ( iter + 1 >= numIters )
                continue;

            const auto e0 = topology.prev( e );
            nextFlipCandidates.set( e0 );
            if ( topology.right( e0 ) )
            {
                nextFlipCandidates.set( topology.prev( e0 ) );
                nextFlipCandidates.set( topology.next( e0.sym() ) );
            }

            const auto e1 = topology.next( e.sym() );
            nextFlipCandidates.set( e1 );
            if ( topology.left( e1 ) )
            {
                nextFlipCandidates.set( topology.next( e1 ) );
                nextFlipCandidates.set( topology.prev( e1.sym() ) );
            }

            const auto e2 = topology.prev( e.sym() );
            nextFlipCandidates.set( e2 );
            if ( topology.right( e2 ) )
            {
                nextFlipCandidates.set( topology.prev( e2 ) );
                nextFlipCandidates.set( topology.next( e2.sym() ) );
            }

            const auto e3 = topology.next( e );
            nextFlipCandidates.set( e3 );
            if ( topology.left( e3 ) )
            {
                nextFlipCandidates.set( topology.next( e3 ) );
                nextFlipCandidates.set( topology.prev( e3.sym() ) );
            }
        }
        if ( flipsDoneBeforeThisIter == flipsDone )
            break;
    }
    return flipsDone;
}

int reduceTotalAngleInMesh( Mesh& mesh, int numIters, const ReduceTotalAngleParams& params, const ProgressCallback& progressCallback )
{
    auto res = reduceTotalAngle( mesh.topology, mesh.points, numIters, params, progressCallback );
    if ( res > 0 )
        mesh.invalidateCaches( false );
    return res;
}

} // namespace MR
