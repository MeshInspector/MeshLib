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
#include "MRAABBTreeObjects.h"
#include "MRInplaceStack.h"
#include "MRSceneRoot.h"

namespace MR
{

Box3f MeshOrPoints::getObjBoundingBox() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) { return mp.mesh.getBoundingBox(); },
        []( const PointCloudPart & pcp ) { return pcp.cloud.getBoundingBox(); }
    }, var_ );
}

void MeshOrPoints::cacheAABBTree() const
{
    std::visit( overloaded{
        []( const MeshPart & mp ) { mp.mesh.getAABBTree(); },
        []( const PointCloudPart & pcp ) { pcp.cloud.getAABBTree(); }
    }, var_ );
}

Box3f MeshOrPoints::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return std::visit( overloaded{
        [toWorld]( const MeshPart & mp ) { return mp.mesh.computeBoundingBox( mp.region, toWorld ); },
        [toWorld]( const PointCloudPart & pcp ) { return pcp.cloud.computeBoundingBox( pcp.region, toWorld ); }
    }, var_ );
}

void MeshOrPoints::accumulate( PointAccumulator& accum, const AffineXf3f* xf ) const
{
    return std::visit( overloaded{
        [&accum, xf]( const MeshPart & mp ) { accumulateFaceCenters( accum, mp, xf ); },
        [&accum, xf]( const PointCloudPart & pcp ) { accumulatePoints( accum, pcp, xf ); }
    }, var_ );
}

std::optional<VertBitSet> MeshOrPoints::pointsGridSampling( float voxelSize, size_t maxVoxels, const ProgressCallback & cb ) const
{
    assert( voxelSize > 0 );
    assert( maxVoxels > 0 );
    auto box = computeBoundingBox();
    if ( !box.valid() )
        return VertBitSet();

    auto bboxDiag = box.size() / voxelSize;
    auto nSamples = bboxDiag[0] * bboxDiag[1] * bboxDiag[2];
    if ( nSamples > maxVoxels )
        voxelSize *= std::cbrt( float(nSamples) / float(maxVoxels) );
    return std::visit( overloaded{
        [voxelSize, cb]( const MeshPart & mp ) { return verticesGridSampling( mp, voxelSize, cb ); },
        [voxelSize, cb]( const PointCloudPart & pcp ) { return pointGridSampling( pcp, voxelSize, cb ); }
    }, var_ );
}

const VertCoords & MeshOrPoints::points() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> const VertCoords & { return mp.mesh.points; },
        []( const PointCloudPart & pcp ) -> const VertCoords & { return pcp.cloud.points; }
    }, var_ );
}

const VertBitSet & MeshOrPoints::validPoints() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> const VertBitSet& { return mp.mesh.topology.getValidVerts(); },
        []( const PointCloudPart & pcp ) -> const VertBitSet& { return pcp.cloud.validPoints; }
    }, var_ );
}

std::function<Vector3f(VertId)> MeshOrPoints::normals() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> std::function<Vector3f(VertId)>
        {
            return [&mesh = mp.mesh]( VertId v ) { return mesh.pseudonormal( v ); };
        },
        []( const PointCloudPart & pcp ) -> std::function<Vector3f(VertId)>
        {
            return !pcp.cloud.hasNormals() ? std::function<Vector3f(VertId)>{} : [&normals = pcp.cloud.normals]( VertId v ) { return normals[v]; };
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
        []( const PointCloudPart & ) { return std::function<float(VertId)>{}; }
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

auto MeshOrPoints::limitedProjector() const -> LimitedProjectorFunc
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> LimitedProjectorFunc
        {
            return [&mp]( const Vector3f & p, ProjectionResult & res )
            {
                MeshProjectionResult mpr = findProjection( p, mp, res.distSq );
                if ( mpr.distSq < res.distSq )
                {
                    res = ProjectionResult
                    {
                        .point = mpr.proj.point,
                        .normal = mp.mesh.pseudonormal( mpr.mtp, mp.region ),
                        .isBd = mpr.mtp.isBd( mp.mesh.topology, mp.region ),
                        .distSq = mpr.distSq,
                        .closestVert = mp.mesh.getClosestVertex( mpr.proj )
                    };
                    return true;
                }
                return false;
            };
        },
        []( const PointCloudPart & pcp ) -> LimitedProjectorFunc
        {
            return [&pcp]( const Vector3f & p, ProjectionResult & res )
            {
                PointsProjectionResult ppr = findProjectionOnPoints( p, pcp, res.distSq );
                if ( ppr.distSq < res.distSq )
                {
                    res = ProjectionResult
                    {
                        .point = pcp.cloud.points[ppr.vId],
                        .normal = ppr.vId < pcp.cloud.normals.size() ? pcp.cloud.normals[ppr.vId] : std::optional<Vector3f>{},
                        .distSq = ppr.distSq,
                        .closestVert = ppr.vId
                    };
                    return true;
                }
                return false;
            };
        }
    }, var_ );
}

std::function<MeshOrPoints::ProjectionResult( const Vector3f& )> MeshOrPointsXf::projector() const
{
    return [lp = limitedProjector()]( const Vector3f & p )
    {
        MeshOrPoints::ProjectionResult res;
        lp( p, res );
        return res;
    };
}

MeshOrPoints::LimitedProjectorFunc MeshOrPointsXf::limitedProjector() const
{
    return [this, f = obj.limitedProjector(), invXf = xf.inverse()]( const Vector3f& p, MeshOrPoints::ProjectionResult& res )
    {
        if ( f( invXf( p ), res ) )
        {
            res.point = xf( res.point );
            if ( res.normal )
                *res.normal = invXf.A.transposed() * *res.normal;
            return true;
        }
        return false;
    };
}

std::optional<MeshOrPoints> getMeshOrPoints( const Object* obj )
{
    if ( auto objMesh = dynamic_cast<const ObjectMeshHolder*>( obj ) )
        return MeshOrPoints( objMesh->meshPart() );
    if ( auto objPnts = dynamic_cast<const ObjectPointsHolder*>( obj ) )
        return MeshOrPoints( objPnts->pointCloudPart() );
    return {};
}

std::optional<MeshOrPointsXf> getMeshOrPointsXf( const Object * obj )
{
    if ( auto objMesh = dynamic_cast<const ObjectMeshHolder*>( obj ) )
        return MeshOrPointsXf{ objMesh->meshPart(), obj->worldXf() };
    if ( auto objPnts = dynamic_cast<const ObjectPointsHolder*>( obj ) )
        return MeshOrPointsXf{ objPnts->pointCloudPart(), obj->worldXf() };
    return {};
}

void projectOnAll(
    const Vector3f& pt,
    const AABBTreeObjects & tree,
    float upDistLimitSq,
    const ProjectOnAllCallback & callback,
    ObjId skipObjId )
{
    if ( tree.nodes().empty() )
        return;

    InplaceStack<NoInitNodeId, 32> subtasks;

    auto addSubTask = [&] ( NodeId n )
    {
        const auto& box = tree.nodes()[n].box;
        if ( !box.valid() )
            return;
        float distSq = box.getDistanceSq( pt );
        if ( distSq < upDistLimitSq )
            subtasks.push( n );
    };

    addSubTask( tree.rootNodeId() );

    while ( !subtasks.empty() )
    {
        const auto n = subtasks.top();
        subtasks.pop();
        const auto node = tree[n];

        if ( node.leaf() )
        {
            auto oid = node.leafId();
            if ( oid == skipObjId )
                continue;
            const auto & obj = tree.obj( oid );

            MeshOrPoints::ProjectionResult pr;
            pr.distSq = upDistLimitSq;
            obj.limitedProjector()( tree.toLocal( oid )( pt ), pr );
            if ( !( pr.distSq < upDistLimitSq ) )
                continue;

            callback( oid, pr );
            continue;
        }

        // first go in left child and then in right
        addSubTask( node.r );
        addSubTask( node.l );
    }
}

MeshOrPoints::ProjectionResult projectWorldPointOntoObjectsRecursive(
    const Vector3f& p,
    const Object* root,
    std::function<bool( const Object& )> projectPred,
    std::function<bool( const Object& )> recursePred
)
{
    MeshOrPoints::ProjectionResult ret;

    auto lambda = [&]( auto& lambda, const Object& cur ) -> void
    {
        if ( !projectPred || projectPred( cur ) )
            getMeshOrPointsXf( &cur )->limitedProjector()( p, ret );

        if ( !recursePred || recursePred( cur ) )
        {
            for ( const auto& child : cur.children() )
                lambda( lambda, *child );
        }
    };
    lambda( lambda, root ? *root : SceneRoot::get() );

    return ret;
}

} // namespace MR
