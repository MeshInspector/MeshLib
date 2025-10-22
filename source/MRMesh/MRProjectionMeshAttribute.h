#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshProject.h"
#include "MRMeshAttributesToUpdate.h"

namespace MR
{

/// this structure contains transformation for projection from one mesh to anther and progress callback
struct ProjectAttributeParams
{
    MeshProjectionTransforms xfs;
    ProgressCallback progressCb;
};

/// projecting the vertex attributes of the old onto the new one
/// returns false if canceled by progress bar
template<typename F>
bool projectVertAttribute( const MeshVertPart& mp, const Mesh& oldMesh, F&& func, const ProjectAttributeParams& params = {} );

/// projecting the face attributes of the old onto the new one
/// returns false if canceled by progress bar
template<typename F>
bool projectFaceAttribute( const MeshPart& mp, const Mesh& oldMesh, F&& func, const ProjectAttributeParams& params = {} );

/// <summary>
/// finds attributes of new mesh by projecting faces/vertices on old mesh
/// \note for now clears edges attributes
/// </summary>
/// <param name="oldMeshData">old mesh along with input attributes</param>
/// <param name="newMeshData">new mesh along with outpuyt attributes</param>
/// <param name="region">optional input region for projecting (usefull if newMesh is changed part of old mesh)</param>
/// <param name="params">parameters of prohecting</param>
[[nodiscard]] MRMESH_API Expected<void> projectObjectMeshData( 
    const ObjectMeshData& oldMeshData, ObjectMeshData& newMeshData, const FaceBitSet* region = nullptr,
    const ProjectAttributeParams& params = {} );

template<typename F>
bool projectVertAttribute( const MeshVertPart& mp, const Mesh& oldMesh, F&& func, const ProjectAttributeParams& params )
{
    return BitSetParallelFor( mp.mesh.topology.getVertIds( mp.region ), [&] ( VertId id )
    {
        auto point = !params.xfs.rigidXfPoint ? mp.mesh.points[id] : ( *params.xfs.rigidXfPoint )( mp.mesh.points[id] );
        auto projectionResult = findProjection( point, oldMesh, FLT_MAX, params.xfs.nonRigidXfTree );
        auto res = projectionResult.mtp;
        VertId v1 = oldMesh.topology.org( res.e );
        VertId v2 = oldMesh.topology.dest( res.e );
        VertId v3 = oldMesh.topology.dest( oldMesh.topology.next( res.e ) );
        func( id, projectionResult, v1, v2, v3 );
    },
    params.progressCb );
}

template<typename F>
bool projectFaceAttribute( const MeshPart& mp, const Mesh& oldMesh, F&& func, const ProjectAttributeParams& params )
{
    return BitSetParallelFor( mp.mesh.topology.getFaceIds( mp.region ), [&] ( FaceId newFaceId )
    {
        auto point = !params.xfs.rigidXfPoint ? mp.mesh.triCenter( newFaceId ) : ( *params.xfs.rigidXfPoint )( mp.mesh.triCenter( newFaceId ) );
        auto projectionResult = findProjection( point, oldMesh, FLT_MAX, params.xfs.nonRigidXfTree );
        func( newFaceId, projectionResult );
    },
    params.progressCb );
}

}
