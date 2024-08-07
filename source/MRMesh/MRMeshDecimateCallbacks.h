#pragma once

#include "MRMesh/MRMesh.h"

namespace MR
{
// there is a vertex parameter that needs to be recalculated when collapsing the edge
template <typename T>
static auto preCollapseVertAttribute( const Mesh& mesh, Vector<T, VertId>& data )
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

// the attribute data of the mesh that needs to be updated
struct MeshAttributesToUpdate
{
    VertUVCoords* uvCoords = nullptr;
    VertColors* colorMap = nullptr;
};

/**
* recalculate texture coordinates and mesh vertex colors for collapsible edges
* usage example
*   MeshAttributesToUpdate meshParams;
*   auto& uvCoords = obj->getUVCoords();
*   auto& colorMap = obj->getVertsColorMap();
*   if ( needUpdateUV )
*       meshParams.uvCoords = &uvCoords;
*   if ( needUpdateColorMap )
*       meshParams.colorMap = &colorMap;
*   auto preCollapse = objectMeshPreCollapseCallback( mesh, meshParams );
*   decimateMesh( mesh, DecimateSettings{ .preCollapse = preCollapse } );
*/
MRMESH_API PreCollapseCallback meshAttributesUpdatePreCollapseCb( const Mesh& mesh, const MeshAttributesToUpdate& params );

}
