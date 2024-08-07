#pragma once

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRVector2.h"

#include <functional>

namespace MR
{

template <typename T>
static std::function<bool( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )> preColapseVertAttribute( const Mesh& mesh, T& data )
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

std::function<bool( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )> creatorPreCollapseFunc( const Mesh& mesh, VertUVCoords& uvCoords, VertColors& colorMap );

}
