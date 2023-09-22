#include "MRMeshOrPoints.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRBox.h"
#include "MRGridSampling.h"

namespace MR
{

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

Box3f MeshOrPoints::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return std::visit( overloaded{
        [toWorld]( const MeshPart & mp ) { return mp.mesh.computeBoundingBox( mp.region, toWorld ); },
        [toWorld]( const PointCloud * pc ) { return pc->computeBoundingBox( toWorld ); }
    }, var_ );
}

std::optional<VertBitSet> MeshOrPoints::pointsGridSampling( float voxelSize, const ProgressCallback & cb )
{
    return std::visit( overloaded{
        [voxelSize, cb]( const MeshPart & mp ) { return verticesGridSampling( mp, voxelSize, cb ); },
        [voxelSize, cb]( const PointCloud * pc ) { return pointGridSampling( *pc, voxelSize, cb ); }
    }, var_ );
}

const VertCoords & MeshOrPoints::points() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> const VertCoords & { return mp.mesh.points; },
        []( const PointCloud * pc ) -> const VertCoords & { return pc->points; }
    }, var_ );
}

} // namespace MR
