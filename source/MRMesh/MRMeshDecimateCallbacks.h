#pragma once

#include "MRMesh.h"
#include "MRMeshAttributesToUpdate.h"

namespace MR
{
// callback that is called when before edge is collapsed 
// (i.e. in decimateMesh) and changes moved vertex attribute to correct value. 
// Useful to update vertices based attributes like uv coordinates or verts colormaps
template <typename T>
auto preCollapseVertAttribute( const Mesh& mesh, Vector<T, VertId>& data );

/**
* Please use this callback when you decimate a mesh with associated data with each vertex
* recalculate texture coordinates and mesh vertex colors for vertices moved during decimation
* usage example
*   MeshAttributesToUpdate meshParams;
*   auto uvCoords = obj->getUVCoords();
*   auto colorMap = obj->getVertsColorMap();
*   if ( needUpdateUV )
*       meshParams.uvCoords = &uvCoords;
*   if ( needUpdateColorMap )
*       meshParams.colorMap = &colorMap;
*   auto preCollapse = meshPreCollapseVertAttribute( mesh, meshParams );
*   decimateMesh( mesh, DecimateSettings{ .preCollapse = preCollapse } );
*/
MRMESH_API PreCollapseCallback meshPreCollapseVertAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params );

template <typename T>
auto preCollapseVertAttribute( const Mesh& mesh, Vector<T, VertId>& data )
{
    auto preCollapse = [&] ( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )
    {
        const auto org = mesh.topology.org( edgeToCollapse );
        const auto dest = mesh.topology.dest( edgeToCollapse );
        const auto orgPos = mesh.orgPnt( edgeToCollapse );
        const auto destPos = mesh.destPnt( edgeToCollapse );

        const auto ab = destPos - orgPos;
        const auto dt = dot( newEdgeOrgPos - orgPos, ab );
        const auto abLengthSq = ab.lengthSq();
        if ( dt <= 0 )
        {
            return true;
        }

        if ( dt >= abLengthSq )
        {
            data[org] = data[dest];
            return true;
        }

        const auto ratio = dt / abLengthSq;
        data[org] = ( 1 - ratio ) * data[org] + ratio * data[dest];

        return true;
    };

    return preCollapse;
}

}
