#include "MRSplineProject.h"
#include "MRMarkedContour.h"
#include "MRMesh.h"
#include "MRSurfacePath.h"
#include "MRSurfaceDistance.h"
#include "MRContour.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

Contour3f projectSpline( const Mesh& mesh, const MarkedContour3f& spline )
{
    MR_TIMER;
    assert( spline.firstLastMarked() );

    const auto sz = spline.contour.size();

    const auto markSeqToPos = makeHashMapWithSeqNums( spline.marks );
    const auto numMarks = markSeqToPos.size();
    assert( markSeqToPos.size() == spline.marks.count() );
    const auto seqToPos = [&markSeqToPos]( size_t seq )
    {
        const auto it = markSeqToPos.find( seq );
        assert( it != markSeqToPos.end() );
        return it->second;
    };

    Contour3f res( sz );
    // regionBetweenMarks[i] is the region between control points markSeqToPos[i] and markSeqToPos[i+1]
    std::vector<FaceBitSet> regionBetweenMarks( numMarks - 1 );
    mesh.getAABBTree(); // prepare before its usage in parallel region
    ParallelFor( size_t( 0 ), numMarks, [&]( size_t seq )
    {
        // project control (marked) points of the spline
        const auto i0 = seqToPos( seq );
        const auto p0 = mesh.projectPoint( spline.contour[i0] );
        res[i0] = p0.proj.point;
        if ( seq + 1 >= numMarks )
            return; // last point

        const auto i1 = seqToPos( seq + 1 );
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
    } );

    // for each point stores the last control (marked) point index,
    // pos2mark[i] = i for control (marked) points, otherwise pos2mark[i] < i
    std::vector<int> pos2mark;
    pos2mark.reserve( sz );
    int lastMark = 0;
    for ( int i = 0; i < sz; ++i )
    {
        if ( spline.marks.test( i ) )
            lastMark = i;
        pos2mark.push_back( lastMark );
    }

    return res;
}

} //namespace MR
