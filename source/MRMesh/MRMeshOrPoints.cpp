#include "MRMeshOrPoints.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRBox.h"
#include "MRGridSampling.h"
#include "MRMeshProject.h"
#include "MRPointsProject.h"
#include "MRObjectMesh.h"
#include "MRObjectPoints.h"
#include "MRBestFit.h"

namespace MR
{

Box3f MeshOrPoints::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return std::visit( overloaded{
        [toWorld]( const MeshPart & mp ) { return mp.mesh.computeBoundingBox( mp.region, toWorld ); },
        [toWorld]( const PointCloud * pc ) { return pc->computeBoundingBox( toWorld ); }
    }, var_ );
}

void MeshOrPoints::accumulate( PointAccumulator& accum, const AffineXf3f* xf ) const
{
    return std::visit( overloaded{
        [&accum, xf]( const MeshPart & mp ) { accumulateFaceCenters( accum, mp, xf ); },
        [&accum, xf]( const PointCloud * pc ) { accumulatePoints( accum, *pc, xf ); }
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
            return !pc->hasNormals() ? std::function<Vector3f(VertId)>{} : [pc]( VertId v ) { return pc->normals[v]; };
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
    return [lp = limitedProjector()]( const Vector3f & p )
    {
        ProjectionResult res;
        lp( p, res );
        return res;
    };
}

auto MeshOrPoints::limitedProjector() const -> std::function<void( const Vector3f & p, ProjectionResult & res )>
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> std::function<void( const Vector3f & p, ProjectionResult & res )>
        {
            return [&mp]( const Vector3f & p, ProjectionResult & res )
            {
                MeshProjectionResult mpr = findProjection( p, mp, res.distSq );
                if ( mpr.distSq < res.distSq )
                    res = ProjectionResult
                    {
                        .point = mpr.proj.point,
                        .normal = mp.mesh.normal( mpr.proj.face ), //mp.mesh.normal( mpr.mtp ) looks more correct here, but it breaks our script test
                        .isBd = mpr.mtp.isBd( mp.mesh.topology ),
                        .distSq = mpr.distSq,
                        .closestVert = mp.mesh.getClosestVertex( mpr.proj )
                    };
            };
        },
        []( const PointCloud * pc ) -> std::function<void( const Vector3f & p, ProjectionResult & res )>
        {
            return [pc]( const Vector3f & p, ProjectionResult & res )
            {
                PointsProjectionResult ppr = findProjectionOnPoints( p, *pc, res.distSq );
                if ( ppr.distSq < res.distSq )
                    res = ProjectionResult
                    {
                        .point = pc->points[ppr.vId],
                        .normal = ppr.vId < pc->normals.size() ? pc->normals[ppr.vId] : std::optional<Vector3f>{},
                        .distSq = ppr.distSq,
                        .closestVert = ppr.vId
                    };
            };
        }
    }, var_ );
}

std::optional<MeshOrPoints> getMeshOrPoints( const VisualObject * obj )
{
    if ( auto objMesh = dynamic_cast<const ObjectMesh*>( obj ) )
        return MeshOrPoints( objMesh->meshPart() );
    if ( auto objPnts = dynamic_cast<const ObjectPoints*>( obj ) )
        return MeshOrPoints( *objPnts->pointCloud() );
    return {};
}

} // namespace MR
