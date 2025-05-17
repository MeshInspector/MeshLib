#include "MRMeshDelone.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRTriMath.h"
#include "MRVector2.h"
#include "MRGeodesicPath.h"
#include "MRTriDist.h"
#include "MREdgeLengthMesh.h"

namespace MR
{

constexpr float NoAngleChangeLimit = 2 * PI_F;

bool checkDeloneQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d, double maxAngleChange )
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
    constexpr double eps = 1e-7; // when we computed in floats then even 1e-5f was too small here and did not prevent infinite loop during resolveMeshDegenerations
    if ( !std::isfinite( metricAC ) )
        return metricAC <= metricBD; // below line returns true if metricAC is +infinity
    return metricAC <= metricBD + eps * ( metricAC + metricBD ); // this shall work even if metricAC and metricBD are infinities, unlike ( metricAC - metricBD ), which becomes NaN
}

bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange )
{
    // the computation of circumcircle diameter in floats for near-degenerate triangles has too large rounding error
    return checkDeloneQuadrangle( Vector3d( a ), Vector3d( b ), Vector3d( c ), Vector3d( d ), double( maxAngleChange ) );
}

FlipEdge canFlipEdge( const MeshTopology & topology, EdgeId edge, const FaceBitSet* region, const UndirectedEdgeBitSet* notFlippable, const VertBitSet* vertRegion )
{
    if ( notFlippable && notFlippable->test( edge ) )
        return FlipEdge::Cannot;

    if ( !topology.isInnerEdge( edge, region ) )
        return FlipEdge::Cannot;

    VertId a, b, c, d;
    topology.getLeftTriVerts( edge, a, c, d );
    assert( a != c );
    b = topology.dest( topology.prev( edge ) );
    if( b == d )
        return FlipEdge::Cannot; // avoid creation of loop edges

    if ( vertRegion )
    {
        if ( !vertRegion->test( a )
          && !vertRegion->test( b )
          && !vertRegion->test( c )
          && !vertRegion->test( d ) )
            return FlipEdge::Cannot;
    }

    bool edgeIsMultiple = false;
    for ( auto e : orgRing0( topology, edge ) )
    {
        if ( topology.dest( e ) == c )
        {
            edgeIsMultiple = true;
            break;
        }
    }

    bool flipEdgeWillBeMultiple = topology.findEdge( b, d ).valid();

    if ( edgeIsMultiple && !flipEdgeWillBeMultiple )
        return FlipEdge::Must;
    if ( !edgeIsMultiple && flipEdgeWillBeMultiple )
        return FlipEdge::Cannot;

    return FlipEdge::Can;
}

bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, const DeloneSettings& settings, float * deviationSqAfterFlip )
{
    return checkDeloneQuadrangleInMesh( mesh.topology, mesh.points, edge, settings, deviationSqAfterFlip );
}

bool checkDeloneQuadrangleInMesh( const MeshTopology & topology, const VertCoords & points, EdgeId edge, const DeloneSettings& settings, float * deviationSqAfterFlip )
{
    const auto can = canFlipEdge( topology, edge, settings.region, settings.notFlippable, settings.vertRegion );
    if ( can == FlipEdge::Cannot )
        return true;
    if ( can == FlipEdge::Must )
        return false;

    VertId a, b, c, d;
    topology.getLeftTriVerts( edge, a, c, d );
    b = topology.dest( topology.prev( edge ) );

    auto ap = points[a];
    auto bp = points[b];
    auto cp = points[c];
    auto dp = points[d];

    if ( deviationSqAfterFlip || settings.maxDeviationAfterFlip < FLT_MAX )
    {
        Vector3f vec, closestOnAC, closestOnBD;
        SegPoints( vec, closestOnAC, closestOnBD,
            ap, cp - ap,   // first diagonal segment
            bp, dp - bp ); // second diagonal segment
        const auto distSq = ( closestOnAC - closestOnBD ).lengthSq();
        if ( deviationSqAfterFlip )
            *deviationSqAfterFlip = distSq;
        if ( distSq > sqr( settings.maxDeviationAfterFlip ) )
            return true; // flipping of given edge will change the surface shape too much
    }

    if ( !isUnfoldQuadrangleConvex( ap, bp, cp, dp ) )
        return true; // cannot flip because 2d quadrangle is concave

    auto maxAngleChange = settings.maxAngleChange;
    if ( settings.criticalTriAspectRatio < FLT_MAX && maxAngleChange < NoAngleChangeLimit )
    {
        const auto maxAspect = std::max( triangleAspectRatio( ap, cp, dp ), triangleAspectRatio( cp, ap, bp ) );
        if ( maxAspect > settings.criticalTriAspectRatio )
        {
            // triangle ACD or ABC is degenerate based on their aspect ratio
            maxAngleChange = NoAngleChangeLimit;
        }
    }

    return checkDeloneQuadrangle( ap, bp, cp, dp, maxAngleChange );
}

bool bestQuadrangleDiagonal( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d )
{
    bool cabcd = isUnfoldQuadrangleConvex( a, b, c, d );
    bool cbcda = isUnfoldQuadrangleConvex( b, c, d, a );
    if ( cabcd != cbcda )
        return cbcda;
    auto metricAC = std::max( circumcircleDiameterSq( a, c, d ), circumcircleDiameterSq( c, a, b ) );
    auto metricBD = std::max( circumcircleDiameterSq( b, d, a ), circumcircleDiameterSq( d, b, c ) );
    return metricAC <= metricBD;
}

void makeDeloneOriginRing( Mesh & mesh, EdgeId e, const DeloneSettings& settings )
{
    mesh.invalidateCaches( false ); // false means that vertex coordinates are not changed
    makeDeloneOriginRing( mesh.topology, mesh.points, e, settings );
}

void makeDeloneOriginRing( MeshTopology& topology, const VertCoords& points, EdgeId e, const DeloneSettings& settings )
{
    topology.flipEdgesIn( e, [&]( EdgeId testEdge )
    {
        return !checkDeloneQuadrangleInMesh( topology, points, testEdge, settings );
    } );
}

int makeDeloneEdgeFlips( Mesh & mesh, const DeloneSettings& settings, int numIters, const ProgressCallback& progressCallback )
{
    int flipsDone = makeDeloneEdgeFlips( mesh.topology, mesh.points, settings, numIters, progressCallback );
    if ( flipsDone > 0 )
        mesh.invalidateCaches( false ); // false means that vertex coordinates are not changed
    return flipsDone;
}

int makeDeloneEdgeFlips( MeshTopology& topology, const VertCoords& points, const DeloneSettings& settings, int numIters, const ProgressCallback& progressCallback )
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
            if ( !checkDeloneQuadrangleInMesh( topology, points, e, settings ) )
                flipCandidates.set( e );
        } );
        nextFlipCandidates.reset();
        int flipsDoneBeforeThisIter = flipsDone;
        for ( UndirectedEdgeId e : flipCandidates )
        {
            if ( checkDeloneQuadrangleInMesh( topology, points, e, settings ) )
                continue;

            ++flipsDone;
            topology.flipEdge( e );
            nextFlipCandidates.set( topology.next( EdgeId( e ) ) );
            nextFlipCandidates.set( topology.prev( EdgeId( e ) ) );
            nextFlipCandidates.set( topology.next( EdgeId( e ).sym() ) );
            nextFlipCandidates.set( topology.prev( EdgeId( e ).sym() ) );
        }
        if ( flipsDoneBeforeThisIter == flipsDone )
            break;
    }
    return flipsDone;
}

int makeDeloneEdgeFlips( EdgeLengthMesh & mesh, const IntrinsicDeloneSettings& settings, int numIters, const ProgressCallback& progressCallback )
{
    if ( numIters <= 0 )
        return 0;
    MR_TIMER;

    auto checkDeloneQuadrangleInMesh = [&]( EdgeId e )
    {
        const auto can = canFlipEdge( mesh.topology, e, settings.region, settings.notFlippable, settings.vertRegion );
        if ( can == FlipEdge::Cannot )
            return true;
        if ( can == FlipEdge::Must )
            return false;
        return mesh.isDelone( e, settings.threshold );
    };

    UndirectedEdgeBitSet flipCandidates( mesh.topology.undirectedEdgeSize() );
    UndirectedEdgeBitSet nextFlipCandidates( mesh.topology.undirectedEdgeSize(), true );

    int flipsDone = 0;
    for ( int iter = 0; iter < numIters; ++iter )
    {
        if ( progressCallback && !progressCallback( float( iter ) / numIters ) )
            return flipsDone;

        flipCandidates.reset();
        BitSetParallelFor( nextFlipCandidates, [&] ( UndirectedEdgeId e )
        {
            if ( !checkDeloneQuadrangleInMesh( e ) )
                flipCandidates.set( e );
        } );
        nextFlipCandidates.reset();
        int flipsDoneBeforeThisIter = flipsDone;
        for ( UndirectedEdgeId e : flipCandidates )
        {
            if ( checkDeloneQuadrangleInMesh( e ) )
                continue;

            if ( !mesh.flipEdge( e ) )
                continue;

            ++flipsDone;
            nextFlipCandidates.set( mesh.topology.next( EdgeId( e ) ) );
            nextFlipCandidates.set( mesh.topology.prev( EdgeId( e ) ) );
            nextFlipCandidates.set( mesh.topology.next( EdgeId( e ).sym() ) );
            nextFlipCandidates.set( mesh.topology.prev( EdgeId( e ).sym() ) );
        }
        if ( flipsDoneBeforeThisIter == flipsDone )
            break;
    }
    return flipsDone;
}

} //namespace MR
