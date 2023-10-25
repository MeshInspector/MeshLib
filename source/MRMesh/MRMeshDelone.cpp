#include "MRMeshDelone.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRTriMath.h"
#include "MRVector2.h"
#include "MRPlanarPath.h"
#include "MRTriDist.h"

namespace MR
{

constexpr float NoAngleChangeLimit = 2 * PI_F;

bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange )
{
    const auto dirABD = dirDblArea( a, b, d );
    const auto dirDBC = dirDblArea( d, b, c );

    if ( dot( dirABD, dirDBC ) < 0 )
        return true; // flipping of given edge will create two faces with opposite normals

    if ( maxAngleChange < NoAngleChangeLimit )
    {
        const auto oldAngle = dihedralAngle( dirABD, dirDBC, d - b );
        const auto dirABC = dirDblArea( a, b, c );
        const auto dirACD = dirDblArea( a, c, d );
        const auto newAngle = dihedralAngle( dirABC, dirACD, a - c );
        const auto angleChange = std::abs( oldAngle - newAngle );
        if ( angleChange > maxAngleChange )
            return true;
    }

    auto metricAC = std::max( circumcircleDiameterSq( a, c, d ), circumcircleDiameterSq( c, a, b ) );
    auto metricBD = std::max( circumcircleDiameterSq( b, d, a ), circumcircleDiameterSq( d, b, c ) );

    // there should be significant difference in metrics (above floating point error) to return false
    constexpr float eps = 1e-5f;
    return metricAC <= metricBD + eps * ( metricAC + metricBD ); // this shall work even if metricAC and metricBD are infinities, unlike ( metricAC - metricBD ), which becomes NaN
}

bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, const DeloneSettings& settings, float * deviationSqAfterFlip )
{
    if ( settings.notFlippable && settings.notFlippable->test( edge.undirected() ) )
        return true; // consider condition satisfied for not-flippable edges

    if ( !mesh.topology.isInnerEdge( edge, settings.region ) )
        return true; // consider condition satisfied for not inner edges

    VertId a, b, c, d;
    mesh.topology.getLeftTriVerts( edge, a, c, d );
    assert( a != c );
    b = mesh.topology.dest( mesh.topology.prev( edge ) );
    if( b == d )
        return true; // consider condition satisfied to avoid creation of loop edges

    bool edgeIsMultiple = false;
    for ( auto e : orgRing0( mesh.topology, edge ) )
    {
        if ( mesh.topology.dest( e ) == c )
        {
            edgeIsMultiple = true;
            break;
        }
    }

    bool flipEdgeWillBeMultiple = mesh.topology.findEdge( b, d ).valid();

    if ( edgeIsMultiple && !flipEdgeWillBeMultiple )
        return false;
    if ( !edgeIsMultiple && flipEdgeWillBeMultiple )
        return true;

    auto ap = mesh.points[a];
    auto bp = mesh.points[b];
    auto cp = mesh.points[c];
    auto dp = mesh.points[d];

    bool degenInputTris = false; // whether triangles ACD and ABC are degenerate based on their aspect ratio
    if ( settings.criticalTriAspectRatio < FLT_MAX &&
         ( deviationSqAfterFlip || settings.maxDeviationAfterFlip < FLT_MAX || settings.maxAngleChange < NoAngleChangeLimit ) )
    {
        const auto maxAspect = std::max( triangleAspectRatio( ap, cp, dp ), triangleAspectRatio( cp, ap, bp ) );
        if ( maxAspect > settings.criticalTriAspectRatio )
            degenInputTris = true;
    }

    if ( deviationSqAfterFlip || settings.maxDeviationAfterFlip < FLT_MAX )
    {
        float distSq = 0;
        if ( !degenInputTris )
        {
            Vector3f vec, closestOnAC, closestOnBD;
            SegPoints( vec, closestOnAC, closestOnBD,
                ap, cp - ap,   // first diagonal segment
                bp, dp - bp ); // second diagonal segment
            distSq = ( closestOnAC - closestOnBD ).lengthSq();
        }
        if ( deviationSqAfterFlip )
            *deviationSqAfterFlip = distSq;
        if ( distSq > sqr( settings.maxDeviationAfterFlip ) )
            return true; // flipping of given edge will change the surface shape too much
    }

    if ( !isUnfoldQuadrangleConvex( ap, bp, cp, dp ) )
        return true; // cannot flip because 2d quadrangle is concave

    return checkDeloneQuadrangle( ap, bp, cp, dp, degenInputTris ? NoAngleChangeLimit : settings.maxAngleChange );
}

int makeDeloneEdgeFlips( Mesh & mesh, const DeloneSettings& settings, int numIters, ProgressCallback progressCallback )
{
    if ( numIters <= 0 )
        return 0;
    MR_TIMER;
    MR_WRITER( mesh );

    UndirectedEdgeBitSet flipCandidates;
    flipCandidates.resize( mesh.topology.undirectedEdgeSize() );
    UndirectedEdgeBitSet nextFlipCandidates;
    nextFlipCandidates.resize( mesh.topology.undirectedEdgeSize(), true );

    int flipsDone = 0;
    for ( int iter = 0; iter < numIters; ++iter )
    {
        if ( progressCallback && !progressCallback( float( iter ) / numIters ) )
            return flipsDone;

        flipCandidates.reset();
        BitSetParallelFor( nextFlipCandidates, [&] ( UndirectedEdgeId e )
        {
            if ( !checkDeloneQuadrangleInMesh( mesh, e, settings ) )
                flipCandidates.set( e );
        } );
        nextFlipCandidates.reset();
        int flipsDoneBeforeThisIter = flipsDone;
        for ( UndirectedEdgeId e : flipCandidates )
        {
            if ( checkDeloneQuadrangleInMesh( mesh, e, settings ) )
                continue;

            mesh.topology.flipEdge( e );
            nextFlipCandidates.set( mesh.topology.next( EdgeId( e ) ).undirected() );
            nextFlipCandidates.set( mesh.topology.prev( EdgeId( e ) ).undirected() );
            nextFlipCandidates.set( mesh.topology.next( EdgeId( e ).sym() ).undirected() );
            nextFlipCandidates.set( mesh.topology.prev( EdgeId( e ).sym() ).undirected() );
            ++flipsDone;
        }
        if ( flipsDoneBeforeThisIter == flipsDone )
            break;
    }
    return flipsDone;
}

void makeDeloneOriginRing( Mesh & mesh, EdgeId e, const DeloneSettings& settings )
{
    MR_WRITER( mesh );
    mesh.topology.flipEdgesIn( e, [&]( EdgeId testEdge )
    {
        return !checkDeloneQuadrangleInMesh( mesh, testEdge, settings );
    } );
}

} //namespace MR
