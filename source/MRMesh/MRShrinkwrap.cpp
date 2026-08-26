#include "MRShrinkwrap.h"
#include "MRMesh.h"
#include "MRMeshProject.h"
#include "MRAffineXf3.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

std::optional<VertCoords> findShrinkwrapPositions( const Mesh & mesh, const Mesh & refMesh,
    const ShrinkwrapParameters & params, const ProgressCallback & cb )
{
    MR_TIMER;

    // converts the coordinates of mesh in the coordinates of refMesh and back
    AffineXf3f meshToRef;
    if ( params.refXf )
        meshToRef = params.refXf->inverse();
    if ( params.xf )
        meshToRef = meshToRef * ( *params.xf );
    const auto refToMesh = meshToRef.inverse();

    VertCoords res = mesh.points;
    const auto & validVerts = mesh.topology.getValidVerts();
    const auto moved = [&] ( VertId v ) { return validVerts.test( v ) && ( !params.region || params.region->test( v ) ); };

    // MeshTriPoint does not depend on the transformations, so the projection is evaluated on refMesh itself
    const auto setPos = [&] ( VertId v, const Vector3f & refPt, const MeshProjectionResult & proj )
    {
        if ( !proj.mtp.e )
            return;
        auto pt = refMesh.triPoint( proj.mtp );
        if ( params.offset != 0 )
        {
            auto dir = ( refPt - pt ).normalized();
            if ( dir == Vector3f{} ) // the vertex is exactly on refMesh, so its own side is unknown
                dir = refMesh.pseudonormal( proj.mtp );
            pt += params.offset * dir;
        }
        res[v] = refToMesh( pt );
    };

    if ( params.projector )
    {
        params.projector->updateMeshData( &refMesh );
        // the projections outside of the region are computed as well to keep one batch call here
        std::vector<MeshProjectionResult> projs( mesh.points.size() );
        params.projector->findProjections( projs, mesh.points.vec_, params.xf, params.refXf, params.upDistLimitSq, params.loDistLimitSq );

        if ( !ParallelFor( 0_v, mesh.points.endId(), [&] ( VertId v )
        {
            if ( moved( v ) )
                setPos( v, meshToRef( mesh.points[v] ), projs[v.get()] );
        }, cb ) )
            return {};
    }
    else
    {
        if ( !ParallelFor( 0_v, mesh.points.endId(), [&] ( VertId v )
        {
            if ( !moved( v ) )
                return;
            const auto refPt = meshToRef( mesh.points[v] );
            setPos( v, refPt, findProjection( refPt, refMesh, params.upDistLimitSq, nullptr, params.loDistLimitSq ) );
        }, cb ) )
            return {};
    }

    return res;
}

bool shrinkwrap( Mesh & mesh, const Mesh & refMesh, const ShrinkwrapParameters & params, const ProgressCallback & cb )
{
    MR_TIMER;
    auto positions = findShrinkwrapPositions( mesh, refMesh, params, cb );
    if ( !positions )
        return false;
    mesh.points = std::move( *positions );
    mesh.invalidateCaches();
    return true;
}

} //namespace MR
