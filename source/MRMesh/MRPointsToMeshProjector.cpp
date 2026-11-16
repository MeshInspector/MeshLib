#include "MRPointsToMeshProjector.h"
#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRMatrix3Decompose.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace
{

/// finds the projection of point on mesh with transformations from createProjectionTransforms;
/// returns distSq in world space and proj.point in the space of the mesh
MeshProjectionResult projectToRefSpace( const Vector3f & point, const Mesh & mesh, const MeshProjectionTransforms & xfs, float upDistLimitSq, float loDistLimitSq )
{
    auto res = findProjection( xfs.rigidXfPoint ? ( *xfs.rigidXfPoint )( point ) : point, mesh, upDistLimitSq, xfs.nonRigidXfTree, loDistLimitSq );
    if ( xfs.nonRigidXfTree && res.valid() )
    {
        // return the point in the space of the mesh as in case of rigid transformation (which is applied to the point instead)
        const auto [v0, v1, v2] = mesh.topology.getTriVerts( res.proj.face );
        res.proj.point = res.tp.interpolate( mesh.points[v0], mesh.points[v1], mesh.points[v2] );
    }
    return res;
}

} // anonymous namespace

void PointsToMeshProjector::updateMeshData( const Mesh* mesh )
{
    mesh_ = mesh;
}

void PointsToMeshProjector::findProjections( std::vector<MeshProjectionResult>& result, const VertCoords& points, const AffineXf3f* objXf, const AffineXf3f* refObjXf, float upDistLimitSq, float loDistLimitSq )
{
    MR_TIMER;
    if ( !mesh_ )
        return;

    result.resize( points.size() );

    AffineXf3f xf;
    auto simplifiedXfs = createProjectionTransforms( xf, objXf, refObjXf );

    ParallelFor( points, [&] ( size_t i )
    {
        result[i] = projectToRefSpace( points[VertId( i )], *mesh_, simplifiedXfs, upDistLimitSq, loDistLimitSq );
    } );
}

size_t PointsToMeshProjector::projectionsHeapBytes( size_t ) const
{
    return 0;
}

VertScalars findSignedDistances(
    const Mesh& refMesh,
    const VertCoords & testPoints, const VertBitSet * validTestPoints,
    const MeshProjectionParameters & params,
    IPointsToMeshProjector * projector )
{
    MR_TIMER;

    AffineXf3f testToRefXf;
    if ( params.refXf )
        testToRefXf = params.refXf->inverse();
    if ( params.xf )
        testToRefXf = testToRefXf * ( *params.xf );

    VertScalars res( testPoints.size(), std::sqrt( params.upDistLimitSq ) );

    if ( projector )
    {
        projector->updateMeshData( &refMesh );
        std::vector<MeshProjectionResult> mpRes( testPoints.size() );
        projector->findProjections( mpRes, testPoints, params.xf, params.refXf, params.upDistLimitSq, params.loDistLimitSq );

        ParallelFor( 0_v, testPoints.endId(), [&] ( VertId v )
        {
            if ( validTestPoints && !validTestPoints->test( v ) )
                return;
            const auto& mpResV = mpRes[v.get()];
            if ( mpResV.valid() )
                res[v] = refMesh.signedDistance( testToRefXf( testPoints[v] ), mpResV );
            else
                res[v] = std::sqrt( mpResV.distSq );
        } );
    }
    else
    {
        // no projector is given, use fully CPU computation with the same transformations as in PointsToMeshProjector
        AffineXf3f storageXf;
        const auto xfs = createProjectionTransforms( storageXf, params.xf, params.refXf );
        ParallelFor( 0_v, testPoints.endId(), [&] ( VertId v )
        {
            if ( validTestPoints && !validTestPoints->test( v ) )
                return;
            const auto mpResV = projectToRefSpace( testPoints[v], refMesh, xfs, params.upDistLimitSq, params.loDistLimitSq );
            if ( mpResV.valid() )
                res[v] = refMesh.signedDistance( testToRefXf( testPoints[v] ), mpResV );
            else
                res[v] = std::sqrt( mpResV.distSq );
        } );
    }
    return res;
}

VertScalars findSignedDistances(
    const Mesh& refMesh,
    const Mesh& mesh,
    const MeshProjectionParameters & params,
    IPointsToMeshProjector * projector )
{
    return findSignedDistances( refMesh, mesh.points, &mesh.topology.getValidVerts(), params, projector );
}

} //namespace MR
