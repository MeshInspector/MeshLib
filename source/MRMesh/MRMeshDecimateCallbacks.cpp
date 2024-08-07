#include "MRMeshDecimateCallbacks.h"

#include "MRMesh/MRVector.h"

namespace MR
{

PreCollapseCallback objectMeshPreCollapseCallback( const Mesh& mesh, const MeshParams& params )
{
    if ( params.uvCoords && params.colorMap )
    {
        auto uvFunc = preColapseVertAttribute( mesh, *params.uvCoords );
        auto colorFunc = preColapseVertAttribute( mesh, *params.colorMap );
        auto preCollapse = [=] ( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )
        {
            bool res = true;
            res = res && uvFunc( edgeToCollapse, newEdgeOrgPos );
            res = res && colorFunc( edgeToCollapse, newEdgeOrgPos );
            return res;
        };

        return preCollapse;
    }

    if ( params.uvCoords )
        return preColapseVertAttribute( mesh, *params.uvCoords );
    if ( params.colorMap )
        return preColapseVertAttribute( mesh, *params.colorMap );

    return {};
}

}