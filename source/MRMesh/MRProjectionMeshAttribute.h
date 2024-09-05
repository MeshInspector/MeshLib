#pragma once

#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshProject.h"
#include "MRMeshAttributesToUpdate.h"

namespace MR
{

// projecting the vertex attributes of the old onto the new one
template<typename F>
void projectVertAttribute( const Mesh& newMesh, const Mesh& oldMesh, F&& func, ProgressCallback progressCb );

// projecting the face attributes of the old onto the new one
template<typename F>
void projectFaceAttribute( const Mesh& newMesh, const Mesh& oldMesh, F&& func, ProgressCallback progressCb );


template<typename F>
void projectVertAttribute( const Mesh& newMesh, const Mesh& oldMesh, F&& func, ProgressCallback progressCb )
{
    BitSetParallelFor( newMesh.topology.getValidVerts(), [&] ( VertId id )
        {
            auto projectionResult = findProjection( newMesh.points[id], oldMesh );
            auto res = projectionResult.mtp;
            VertId v1 = oldMesh.topology.org( res.e );
            VertId v2 = oldMesh.topology.dest( res.e );
            VertId v3 = oldMesh.topology.dest( oldMesh.topology.next( res.e ) );
            func( id, projectionResult, v1, v2, v3 );
        },
    progressCb );
}

template<typename F>
void projectFaceAttribute( const Mesh& newMesh, const Mesh& oldMesh, F&& func, ProgressCallback progressCb )
{
    BitSetParallelFor( newMesh.topology.getValidFaces(), [&] ( FaceId newFaceId )
    {
        auto point = newMesh.triCenter( newFaceId );
        auto projectionResult = findProjection( point, oldMesh );
        func( newFaceId, projectionResult );
    },
    progressCb );
}

}
