#pragma once

#include "MRMesh.h"
#include "MRMeshAttributesToUpdate.h"

namespace MR
{
// callback that is called each time edge (e) is split into (e1->e), but before the ring is made Delone
// (i.e. in subdivideMesh) and changes moved vertex attribute to correct value.
// Useful to update vertices based attributes like uv coordinates or verts colormaps
template <typename T>
auto onEdgeSplitVertAttribute( const Mesh& mesh, Vector<T, VertId>& data );

// callback that is called each time edge (e) is split into (e1->e), but before the ring is made Delone
// (i.e. in subdivideMesh) and changes moved vertex attribute to correct value.
// Useful to update face based attributes like texturePerFace or face colors
template <typename T>
auto onEdgeSplitFaceAttribute( const Mesh& mesh, Vector<T, FaceId>& data );

/**
* auto uvCoords = obj_->getUVCoords();
* auto texturePerFace = obj_->getTexturePerFace();
* MeshAttributesToUpdate meshParams;
* if ( !uvCoords.empty() )
*     meshParams.uvCoords = &uvCoords;
* if ( !texturePerFace.empty() )
*     meshParams.texturePerFace = &texturePerFace;
* subs.onEdgeSplit = meshOnEdgeSplitAttribute( *obj_->varMesh(), meshParams );
* subdivideMesh( *obj_->varMesh(), subs );
*/
MRMESH_API OnEdgeSplit meshOnEdgeSplitAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params );

MRMESH_API OnEdgeSplit meshOnEdgeSplitVertAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params );

MRMESH_API OnEdgeSplit meshOnEdgeSplitFaceAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params );

template <typename T>
auto onEdgeSplitVertAttribute( const Mesh& mesh, Vector<T, VertId>& data )
{
    auto onEdgeSplit = [&] ( EdgeId e1, EdgeId e )
    {
        const auto org = mesh.topology.org( e1 );
        const auto dest = mesh.topology.dest( e );
        if ( org < data.size() && dest < data.size() )
            data.push_back( data[org] * 0.5f + data[dest] * 0.5f ); // important to multiply each of ends individually not to lose data for clamped integer values (like colors)
    };

    return onEdgeSplit;
}

template <typename T>
auto onEdgeSplitFaceAttribute( const Mesh& mesh, Vector<T, FaceId>& data )
{
    auto onEdgeSplit = [&] ( EdgeId e1, EdgeId e )
    {
        // getting a left face for an edge that has been split.
        auto oldLeft = mesh.topology.left( e );
        // getting a left face for a new edge that was added during splitting.
        auto newLeft = mesh.topology.left( e1 );

        if ( newLeft && oldLeft && oldLeft < data.size() )
            data.autoResizeSet( newLeft, data[oldLeft] );

        // getting a right face for an edge that has been split.
        auto oldRight = mesh.topology.right( e );
        // getting a right face for a new edge that was added during splitting.
        auto newRight = mesh.topology.right( e1 );

        if ( newRight && oldRight && oldRight < data.size() )
            data.autoResizeSet( newRight, data[oldRight] );
    };

    return onEdgeSplit;
}

}
