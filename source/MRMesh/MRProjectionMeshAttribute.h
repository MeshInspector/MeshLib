#pragma once

#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshProject.h"
#include "MRMeshAttributesToUpdate.h"

namespace MR
{

// projecting the vertex attributes of the old onto the new one
// returns false if canceled by progress bar
template<typename F>
bool projectVertAttribute( const MeshVertPart& mp, const Mesh& oldMesh, F&& func, ProgressCallback progressCb );

// projecting the face attributes of the old onto the new one
// returns false if canceled by progress bar
template<typename F>
bool projectFaceAttribute( const MeshPart& mp, const Mesh& oldMesh, F&& func, ProgressCallback progressCb );


template<typename F>
bool projectVertAttribute( const MeshVertPart& mp, const Mesh& oldMesh, F&& func, ProgressCallback progressCb )
{
    return BitSetParallelFor( mp.mesh.topology.getVertIds( mp.region ), [&] ( VertId id )
        {
            auto projectionResult = findProjection( mp.mesh.points[id], oldMesh );
            auto res = projectionResult.mtp;
            VertId v1 = oldMesh.topology.org( res.e );
            VertId v2 = oldMesh.topology.dest( res.e );
            VertId v3 = oldMesh.topology.dest( oldMesh.topology.next( res.e ) );
            func( id, projectionResult, v1, v2, v3 );
        },
    progressCb );
}

template<typename F>
bool projectFaceAttribute( const MeshPart& mp, const Mesh& oldMesh, F&& func, ProgressCallback progressCb )
{
    return BitSetParallelFor( mp.mesh.topology.getFaceIds( mp.region ), [&] ( FaceId newFaceId )
    {
        auto projectionResult = findProjection( mp.mesh.triCenter( newFaceId ), oldMesh );
        func( newFaceId, projectionResult );
    },
    progressCb );
}

}
