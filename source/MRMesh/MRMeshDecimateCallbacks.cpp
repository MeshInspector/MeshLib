#include "MRMeshDecimateCallbacks.h"

#include "MRMesh/MRVector.h"

namespace MR
{

std::function<bool( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )> creatorPreCollapseFunc( const Mesh & mesh, VertUVCoords & uvCoords, VertColors & colorMap )
{
    if ( !uvCoords.empty() )
        return preColapseVertAttribute( mesh, uvCoords );
    if ( !colorMap.empty() )
        return preColapseVertAttribute( mesh, colorMap );

    auto uvFunc = preColapseVertAttribute( mesh, uvCoords );
    auto colorFunc = preColapseVertAttribute( mesh, colorMap );
    auto preCollapse = [=] ( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )
    {
        bool res = true;
        res = res && uvFunc( edgeToCollapse, newEdgeOrgPos );
        res = res && colorFunc( edgeToCollapse, newEdgeOrgPos );
        return res;
    };

    return preCollapse;
}

}