#include "MRPointsToMeshProjector.h"
#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRMatrix3Decompose.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

void PointsToMeshProjector::updateMeshData( const Mesh* mesh )
{
    mesh_ = mesh;
}

void PointsToMeshProjector::findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, const AffineXf3f* objXf, const AffineXf3f* refObjXf, float upDistLimitSq, float loDistLimitSq )
{
    MR_TIMER
    if ( !mesh_ )
    
    return;
    
    result.resize( points.size() );

    const AffineXf3f* notRigidRefXf{ nullptr };
    if ( refObjXf && !isRigid( refObjXf->A ) )
        notRigidRefXf = refObjXf;

    AffineXf3f xf;
    const AffineXf3f* xfPtr{ nullptr };
    if ( notRigidRefXf || !refObjXf )
        xfPtr = objXf;
    else
    {
        xf = refObjXf->inverse();
        if ( objXf )
            xf = xf * ( *objXf );
        xfPtr = &xf;
    }

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, points.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
            result[i] = findProjection( xfPtr ? ( *xfPtr )( points[VertId( i )] ) : points[VertId( i )], *mesh_, upDistLimitSq, notRigidRefXf, loDistLimitSq );
    } );
}

size_t PointsToMeshProjector::projectionsHeapBytes( size_t ) const
{
    return 0;
}

}