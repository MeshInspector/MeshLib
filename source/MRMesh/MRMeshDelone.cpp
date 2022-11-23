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

inline auto dir( const auto& p, const auto& q, const auto& r )
{
    return cross( q - p, r - p );
}
inline auto area( const auto& p, const auto& q, const auto& r )
{
    return dir( p, q, r ).length();
}

bool checkDeloneQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d, double maxAngleChange )
{
    const auto dirABD = dir( a, b, d );
    const auto dirDBC = dir( d, b, c );

    if ( dot( dirABD, dirDBC ) < 0 )
        return true; // flipping of given edge will create two faces with opposite normals

    if ( maxAngleChange < NoAngleChangeLimit )
    {
        const auto oldAngle = dihedralAngle( dirABD, dirDBC, d - b );
        const auto dirABC = dir( a, b, c );
        const auto dirACD = dir( a, c, d );
        const auto newAngle = dihedralAngle( dirABC, dirACD, a - c );
        const auto angleChange = std::abs( oldAngle - newAngle );
        if ( angleChange > maxAngleChange )
            return true;
    }

    auto metricAC = std::max( circumcircleDiameter( a, c, d ), circumcircleDiameter( c, a, b ) );
    auto metricBD = std::max( circumcircleDiameter( b, d, a ), circumcircleDiameter( d, b, c ) );
    return metricAC <= metricBD;
}

bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange )
{
    return checkDeloneQuadrangle( Vector3d{a}, Vector3d{b}, Vector3d{c}, Vector3d{d}, maxAngleChange );
}

template<typename T>
bool checkAspectRatiosInQuadrangleT( const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c, const Vector3<T>& d, T maxAngleChange, T criticalTriAspectRatio )
{
    auto metricAC = std::max( triangleAspectRatio( a, c, d ), triangleAspectRatio( c, a, b ) );
    auto metricBD = std::max( triangleAspectRatio( b, d, a ), triangleAspectRatio( d, b, c ) );
    if ( metricAC <= metricBD )
        return true;
    if ( metricAC < criticalTriAspectRatio && maxAngleChange < NoAngleChangeLimit )
    {
        const auto dirABD = dir( a, b, d );
        const auto dirDBC = dir( d, b, c );
        const auto oldAngle = dihedralAngle( dirABD, dirDBC, d - b );

        const auto dirABC = dir( a, b, c );
        const auto dirACD = dir( a, c, d );
        const auto newAngle = dihedralAngle( dirABC, dirACD, a - c );

        const auto angleChange = std::abs( oldAngle - newAngle );
        if ( angleChange > maxAngleChange )
            return true;
    }
    else if ( metricAC >= criticalTriAspectRatio )
    {
        const auto sABC = area( a, b, c );
        const auto sACD = area( a, c, d );

        const auto sABD = area( a, b, d );
        const auto sDBC = area( d, b, c );

        // in case of degenerate triangles, select the subdivision with smaller total area
        if ( sABC + sACD < sABD + sDBC )
            return true;
    }
    return false;
}

bool checkAspectRatiosInQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d, double maxAngleChange, double criticalTriAspectRatio )
{
    return checkAspectRatiosInQuadrangleT( a, b, c, d, maxAngleChange, criticalTriAspectRatio );
}

bool checkAspectRatiosInQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange, float criticalTriAspectRatio )
{
    return checkAspectRatiosInQuadrangleT( a, b, c, d, maxAngleChange, criticalTriAspectRatio );
}

bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, const DeloneSettings& settings )
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

    bool flipEdgeWillBeMultiple = false;
    for ( auto e : orgRing( mesh.topology, mesh.topology.next( edge ).sym()  ) )
    {
        assert( mesh.topology.org( e ) == d );
        if ( mesh.topology.dest( e ) == b )
        {
            flipEdgeWillBeMultiple = true;
            break;
        }
    }

    if ( edgeIsMultiple && !flipEdgeWillBeMultiple )
        return false;
    if ( !edgeIsMultiple && flipEdgeWillBeMultiple )
        return true;

    auto ap = mesh.points[a];
    auto bp = mesh.points[b];
    auto cp = mesh.points[c];
    auto dp = mesh.points[d];

    if ( settings.maxDeviationAfterFlip < FLT_MAX )
    {
        Vector3f vec, closestOnAC, closestOnBD;
        SegPoints( vec, closestOnAC, closestOnBD,
            ap, cp - ap,   // first diagonal segment
            bp, dp - bp ); // second diagonal segment
        double distSq = ( closestOnAC - closestOnBD ).lengthSq();
        if ( distSq > sqr( settings.maxDeviationAfterFlip ) )
            return true; // flipping of given edge will change the surface shape too much
    }

    if ( !isUnfoldQuadrangleConvex( ap, bp, cp, dp ) )
        return true; // cannot flip because 2d quadrangle is concave

    if ( settings.criticalTriAspectRatio < FLT_MAX )
        return checkAspectRatiosInQuadrangle( ap, bp, cp, dp, settings.maxAngleChange, settings.criticalTriAspectRatio );
    else
        return checkDeloneQuadrangle( ap, bp, cp, dp, settings.maxAngleChange );
}

int makeDeloneEdgeFlips( Mesh & mesh, const DeloneSettings& settings, int numIters, ProgressCallback progressCallback )
{
    if ( numIters <= 0 )
        return 0;
    MR_TIMER;
    MR_WRITER( mesh );

    int flipsDone = 0;
    for ( int iter = 0; iter < numIters; ++iter )
    {
        if ( progressCallback && !progressCallback( float( iter ) / numIters ) )
            return flipsDone;

        UndirectedEdgeBitSet flipCandidates;
        flipCandidates.resize( mesh.topology.undirectedEdgeSize() );
        BitSetParallelForAll( flipCandidates, [&] ( UndirectedEdgeId e )
        {
            flipCandidates.set( e, !checkDeloneQuadrangleInMesh( mesh, e, settings ) );
        } );
        int flipsDoneBeforeThisIter = flipsDone;
        for ( UndirectedEdgeId e : flipCandidates )
        {
            if ( checkDeloneQuadrangleInMesh( mesh, e, settings ) )
                continue;

            mesh.topology.flipEdge( e );
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
    const EdgeId e0 = e;
    for (;;)
    {
        auto testEdge = mesh.topology.prev( e.sym() );
        if ( checkDeloneQuadrangleInMesh( mesh, testEdge, settings ) )
        {
            e = mesh.topology.next( e );
            if ( e == e0 )
                break; // full ring has been inspected
            continue;
        }
        mesh.topology.flipEdge( testEdge );
    } 
}

} //namespace MR
