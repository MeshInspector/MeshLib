#include "MRMeshDecimateCallbacks.h"

#include "MRMesh/MRVector.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRColor.h"

namespace MR
{

PreCollapseCallback meshPreCollapseVertAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params )
{
    if ( params.uvCoords && params.colorMap )
    {
        auto uvFunc = preCollapseVertAttribute( mesh, *params.uvCoords );
        auto colorFunc = preCollapseVertAttribute( mesh, *params.colorMap );
        auto preCollapse = [=] ( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )
        {
            uvFunc( edgeToCollapse, newEdgeOrgPos );
            colorFunc( edgeToCollapse, newEdgeOrgPos );
            return true;
        };

        return preCollapse;
    }

    if ( params.uvCoords )
        return preCollapseVertAttribute( mesh, *params.uvCoords );
    if ( params.colorMap )
        return preCollapseVertAttribute( mesh, *params.colorMap );

    return {};
}

}
