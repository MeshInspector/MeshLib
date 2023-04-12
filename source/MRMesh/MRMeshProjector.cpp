#include "MRMeshProjector.h"
#include "MRMesh.h"
#include "MRPch/MRTBB.h"
#include "MRAffineXf3.h"
#include <chrono>

namespace MR
{

void MeshProjector::updateMeshData( std::shared_ptr<const Mesh> mesh )
{
    mesh_ = mesh;
}

std::vector<MeshProjectionResult> MeshProjector::findProjections( const std::vector<Vector3f>& points, const AffineXf3f* xf, const AffineXf3f* refXfPtr, float upDistLimitSq, float loDistLimitSq )
{
    const auto start = std::chrono::steady_clock::now();    
    
    std::vector<MeshProjectionResult> projResults( points.size() );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, points.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
            projResults[i] = findProjection( xf ? (*xf)( points[VertId( i )] ) : points[VertId( i )], *mesh_, upDistLimitSq, refXfPtr, loDistLimitSq );
    } );
    
    const auto duration = std::chrono::steady_clock::now() - start;

    return projResults;
}


}