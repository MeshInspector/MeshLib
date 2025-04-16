#include "MRPointsToMeshProjector.h"
#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRMatrix3Decompose.h"
#include "MRParallelFor.h"
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
    MR_TIMER;
    if ( !mesh_ )
        return;

    result.resize( points.size() );

    AffineXf3f xf;
    auto simplifiedXfs = createProjectionTransforms( xf, objXf, refObjXf );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, points.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
            result[i] = findProjection( simplifiedXfs.rigidXfPoint ? ( *simplifiedXfs.rigidXfPoint )( points[VertId( i )] ) : points[VertId( i )], *mesh_, upDistLimitSq, simplifiedXfs.nonRigidXfTree, loDistLimitSq );
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
        projector->findProjections( mpRes, testPoints.vec_, params.xf, params.refXf, params.upDistLimitSq, params.loDistLimitSq );

        ParallelFor( 0_v, testPoints.endId(), [&] ( VertId v )
        {
            if ( validTestPoints && !validTestPoints->test( v ) )
                return;
            const auto& mpResV = mpRes[v.get()];
            if ( mpResV.mtp.e )
                res[v] = refMesh.signedDistance( testToRefXf( testPoints[v] ), mpResV );
            else
                res[v] = std::sqrt( mpResV.distSq );
        } );
    }
    else
    {
        // no projector is given, use fully CPU computation
        ParallelFor( 0_v, testPoints.endId(), [&] ( VertId v )
        {
            if ( validTestPoints && !validTestPoints->test( v ) )
                return;
            const auto pt = testToRefXf( testPoints[v] );
            const auto mpResV = findProjection( pt, refMesh, params.upDistLimitSq, nullptr, params.loDistLimitSq );
            if ( mpResV.mtp.e )
                res[v] = refMesh.signedDistance( pt, mpResV );
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
