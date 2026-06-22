#include "MRSplineProject.h"
#include "MRMarkedContour.h"
#include "MRMesh.h"
#include "MRSurfacePath.h"
#include "MRSurfaceDistance.h"
#include "MRContour.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

namespace
{

FaceBitSet facesWithinRange( const MeshTopology& topology, const VertScalars& vertDist, float range )
{
    MR_TIMER;
    FaceBitSet res( topology.faceSize() );
    BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f )
    {
        auto vs = topology.getTriVerts( f );
        for ( VertId v : vs )
            if ( vertDist[v] <= range )
            {
                res.set( f );
                break;
            }
    } );
    return res;
}

} //anonymous namespace

std::vector<MeshTriPoint> projectSplineAsMTP( const Mesh& mesh, const MarkedContour3f& spline )
{
    MR_TIMER;
    assert( spline.firstLastMarked() );

    const auto sz = spline.contour.size();

    std::vector<size_t> markSeqToPos;
    markSeqToPos.reserve( spline.marks.count() );
    for ( auto m : spline.marks )
        markSeqToPos.push_back( m );
    const auto numMarks = markSeqToPos.size();
    assert( markSeqToPos.size() == spline.marks.count() );

    std::vector<MeshTriPoint> res( sz );
    // regionBetweenMarks[i] is the region between control points markSeqToPos[i] and markSeqToPos[i+1]
    std::vector<FaceBitSet> regionBetweenMarks( numMarks - 1 );
    mesh.getAABBTree(); // prepare before its usage in parallel region
    ParallelFor( size_t( 0 ), numMarks, [&]( size_t seq )
    {
        // project control (marked) points of the spline
        const auto i0 = markSeqToPos[seq];
        const auto p0 = mesh.projectPoint( spline.contour[i0] );
        res[i0] = p0.mtp;
        if ( seq + 1 >= numMarks )
            return; // last point

        const auto i1 = markSeqToPos[seq + 1];
        if ( i0 + 1 == i1 )
            return; // zero not-control points in between
        const auto p1 = mesh.projectPoint( spline.contour[i1] );

        auto maybeSurfPath = computeGeodesicPath( mesh, p0.mtp, p1.mtp );
        assert( maybeSurfPath );
        if ( !maybeSurfPath )
            return;

        const GeodesicPath geoPath{ .start = p0.mtp, .mids = std::move( *maybeSurfPath ), .end = p1.mtp };
        const auto geoPathAsContour = geodesicPathToContour3f( mesh, geoPath );
        const auto geoPathLen = calcLength( geoPathAsContour );
        const auto midPointId = findContourPointByLength( geoPathAsContour, geoPathLen / 2 );
        const auto surfDist = computeSurfaceDistances( mesh, geoPath[midPointId], geoPathLen / 2 );
        regionBetweenMarks[seq] = facesWithinRange( mesh.topology, surfDist, geoPathLen / 2 );
        assert( regionBetweenMarks[seq].any() );
    } );

    // for each point stores the last control (marked) point index,
    // pos2mark[i] = i for control (marked) points, otherwise pos2mark[i] < i
    std::vector<int> pos2mark;
    pos2mark.reserve( sz );
    int lastMark = -1;
    for ( int i = 0; i < sz; ++i )
    {
        if ( spline.marks.test( i ) )
            ++lastMark;
        pos2mark.push_back( lastMark );
    }

    // project not control points on appropriate parts of the mesh
    ParallelFor( size_t( 0 ), sz, [&]( size_t i )
    {
        if ( spline.marks.test( i ) )
            return; // control point - already projected above

        const auto& faceRegion = regionBetweenMarks[pos2mark[i]];
        const auto p = mesh.projectPoint( spline.contour[i], FLT_MAX, &faceRegion );
        res[i] = p.mtp;
    } );

    return res;
}

Contour3f projectSpline( const Mesh& mesh, const MarkedContour3f& spline )
{
    MR_TIMER;
    auto mtps = projectSplineAsMTP( mesh, spline );
    Contour3f res;
    resizeNoInit( res, mtps.size() );
    ParallelFor( res, [&]( size_t i )
    {
        res[i] = mesh.triPoint( mtps[i] );
    } );
    return res;
}

} //namespace MR
