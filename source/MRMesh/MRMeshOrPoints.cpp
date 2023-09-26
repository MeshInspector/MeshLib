#include "MRMeshOrPoints.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRBox.h"
#include "MRGridSampling.h"
#include "MRMeshProject.h"
#include "MRPointsProject.h"

namespace MR
{

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

std::function<Vector3f(VertId)> MeshOrPoints::normals() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> std::function<Vector3f(VertId)>
        {
            return [&mesh = mp.mesh]( VertId v ) { return mesh.normal( v ); };
        },
        []( const PointCloud * pc ) -> std::function<Vector3f(VertId)>
        { 
            return pc->normals.empty() ? std::function<Vector3f(VertId)>{} : [pc]( VertId v ) { return pc->normals[v]; };
        }
    }, var_ );
}

std::function<float(VertId)> MeshOrPoints::weights() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> std::function<float(VertId)>
        {
            return [&mesh = mp.mesh]( VertId v ) { return mesh.dblArea( v ); };
        },
        []( const PointCloud * ) { return std::function<float(VertId)>{}; }
    }, var_ );
}

auto MeshOrPoints::projector() const -> std::function<ProjectionResult( const Vector3f & )>
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> std::function<ProjectionResult( const Vector3f & )>
        {
            return [&mp]( const Vector3f & p )
            {
                MeshProjectionResult mpr = findProjection( p, mp );
                return ProjectionResult
                {
                    .point = mpr.proj.point,
                    .normal = mp.mesh.normal( mpr.mtp ),
                    .isBd = mpr.mtp.isBd( mp.mesh.topology ),
                    .distSq = mpr.distSq
                };
            };
        },
        []( const PointCloud * pc ) -> std::function<ProjectionResult( const Vector3f & )>
        {
            return [pc]( const Vector3f & p )
            {
                PointsProjectionResult ppr = findProjectionOnPoints( p, *pc );
                return ProjectionResult
                {
                    .point = pc->points[ppr.vId],
                    .normal = ppr.vId < pc->normals.size() ? pc->normals[ppr.vId] : std::optional<Vector3f>{},
                    .distSq = ppr.distSq
                };
            };
        }
    }, var_ );
}

} // namespace MR
