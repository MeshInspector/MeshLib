#include "MRSplineProject.h"
#include "MRMarkedContour.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

Contour3f projectSpline( const Mesh& mesh, const MarkedContour3f& spline )
{
    MR_TIMER;
    assert( spline.firstLastMarked() );

    const auto markSeqToPos = makeHashMapWithSeqNums( spline.marks );
    assert( markSeqToPos.size() == spline.marks.count() );

    Contour3f res = spline.contour;

    // project control (marked) points of the spline
    ParallelFor( size_t( 0 ), markSeqToPos.size(), [&]( size_t seq )
    {
        const auto it = markSeqToPos.find( seq );
        assert( it != markSeqToPos.end() );
        const auto i = it->second;
        res[i] = mesh.projectPoint( res[i] ).proj.point;
    } );

    return res;
}

} //namespace MR
