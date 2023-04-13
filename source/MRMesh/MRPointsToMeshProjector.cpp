#include "MRPointsToMeshProjector.h"
#include "MRMesh.h"
#include "MRPch/MRTBB.h"
#include "MRAffineXf3.h"
#include "MRMatrix3Decompose.h"

namespace MR
{

void PointsToMeshProjector::updateMeshData( std::shared_ptr<const Mesh> mesh )
{
    mesh_ = mesh;
}

void PointsToMeshProjector::findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, const AffineXf3f& objXf, const AffineXf3f& refObjXf, float upDistLimitSq, float loDistLimitSq )
{
    result.resize( points.size() );

    refXfPtr_ = isRigid( refObjXf.A ) ? nullptr : &refObjXf;
    xf_ = refXfPtr_ ? objXf : refObjXf.inverse() * objXf;

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, points.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
            result[i] = findProjection( ( xf_ != AffineXf3f{} ) ? ( xf_ )( points[VertId( i )] ) : points[VertId( i )], * mesh_, upDistLimitSq, refXfPtr_, loDistLimitSq );
    } );
}


}