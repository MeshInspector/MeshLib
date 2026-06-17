#include "MRSplineProject.h"
#include "MRMarkedContour.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

Contour3f projectSpline( const Mesh& mesh, const MarkedContour3f& spline )
{
    MR_TIMER;
    assert( spline.firstLastMarked() );

    Contour3f res = spline.contour;

    BitSetParallelFor( spline.marks, [&]( size_t i )
    {
        res[i] = mesh.projectPoint( res[i] ).proj.point;
    } );

    return res;
}

} //namespace MR
