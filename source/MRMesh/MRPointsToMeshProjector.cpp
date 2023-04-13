#include "MRPointsToMeshProjector.h"
#include "MRMesh.h"
#include "MRPch/MRTBB.h"
#include "MRAffineXf3.h"
#include "MRMatrix3Decompose.h"
#include <chrono>

namespace MR
{

void PointsToMeshProjector::updateMeshData( std::shared_ptr<const Mesh> mesh )
{
    mesh_ = mesh;
}

void PointsToMeshProjector::updateTransforms( const AffineXf3f& objXf, const AffineXf3f& refObjXf )
{    
    refXfPtr_ = isRigid( refObjXf.A ) ? nullptr : const_cast<AffineXf3f*>( &refObjXf );
    xf_ = refXfPtr_ ? objXf : refObjXf.inverse() * objXf;
}

std::vector<MeshProjectionResult> PointsToMeshProjector::findProjections( const std::vector<Vector3f>& points, float upDistLimitSq, float loDistLimitSq )
{
    const auto start = std::chrono::steady_clock::now();
    
    std::vector<MeshProjectionResult> projResults( points.size() );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, points.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
            projResults[i] = findProjection( ( xf_ != AffineXf3f{} ) ? ( xf_ )( points[VertId( i )] ) : points[VertId( i )], * mesh_, upDistLimitSq, refXfPtr_, loDistLimitSq );
    } );
    
    const auto duration = std::chrono::steady_clock::now() - start;

    return projResults;
}


}