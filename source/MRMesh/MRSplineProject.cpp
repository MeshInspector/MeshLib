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
    const auto numMarks = markSeqToPos.size();
    assert( markSeqToPos.size() == spline.marks.count() );
    const auto seqToPos = [&markSeqToPos]( size_t seq )
    {
        const auto it = markSeqToPos.find( seq );
        assert( it != markSeqToPos.end() );
        return it->second;
    };

    Contour3f res = spline.contour;

    // project control (marked) points of the spline
    ParallelFor( size_t( 0 ), numMarks, [&]( size_t seq )
    {
        const auto i = seqToPos( seq );
        res[i] = mesh.projectPoint( res[i] ).proj.point;
    } );

    if ( numMarks <= 1 )
    {
        assert( res.size() == 1 );
        return res;
    }

    // regionBetweenMarks[i] is the region between control points markSeqToPos[i] and markSeqToPos[i+1]
    std::vector<FaceBitSet> regionBetweenMarks( numMarks - 1 );
    ParallelFor( size_t( 0 ), numMarks - 1, [&]( size_t seq )
    {
        const auto i0 = seqToPos( seq );
        const auto i1 = seqToPos( seq + 1 );
    } );

    return res;
}

} //namespace MR
