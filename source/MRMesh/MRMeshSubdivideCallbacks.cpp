#include "MRMeshSubdivideCallbacks.h"

#include "MRMesh/MRColor.h"

namespace MR
{

OnEdgeSplit meshOnEdgeSplitAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params )
{
    auto uvFunc = onEdgeSplitVertAttribute( mesh, *params.uvCoords );
    auto colorFunc = onEdgeSplitVertAttribute( mesh, *params.colorMap );
    auto texturePerFaceFunc = onEdgeSplitFaceAttribute( mesh, *params.texturePerFace );
    auto faceColorsFunc = onEdgeSplitFaceAttribute( mesh, *params.faceColors );

    auto preCollapse = [=] ( EdgeId e1, EdgeId e )
    {
        if ( params.uvCoords )
            uvFunc( e1, e );
        if ( params.colorMap )
            colorFunc( e1, e );
        if ( params.texturePerFace )
            texturePerFaceFunc( e1, e );
        if ( params.faceColors )
            faceColorsFunc( e1, e );
        return true;
    };

    return preCollapse;
}

OnEdgeSplit meshOnEdgeSplitVertAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params )
{
    if ( params.uvCoords && params.colorMap )
    {
        auto uvFunc = onEdgeSplitVertAttribute( mesh, *params.uvCoords );
        auto colorFunc = onEdgeSplitVertAttribute( mesh, *params.colorMap );
        auto preCollapse = [=] ( EdgeId e1, EdgeId e )
        {
            uvFunc( e1, e );
            colorFunc( e1, e );
            return true;
        };

        return preCollapse;
    }

    if ( params.uvCoords )
        return onEdgeSplitVertAttribute( mesh, *params.uvCoords );
    if ( params.colorMap )
        return onEdgeSplitVertAttribute( mesh, *params.colorMap );

    return {};
}

OnEdgeSplit meshOnEdgeSplitFaceAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params )
{
    if ( params.texturePerFace && params.faceColors )
    {
        auto texturePerFaceFunc = onEdgeSplitFaceAttribute( mesh, *params.texturePerFace );
        auto faceColorsFunc = onEdgeSplitFaceAttribute( mesh, *params.faceColors );
        auto preCollapse = [=] ( EdgeId e1, EdgeId e )
        {
            texturePerFaceFunc( e1, e );
            faceColorsFunc( e1, e );
            return true;
        };

        return preCollapse;
    }

    if ( params.uvCoords )
        return onEdgeSplitFaceAttribute( mesh, *params.texturePerFace );
    if ( params.colorMap )
        return onEdgeSplitFaceAttribute( mesh, *params.faceColors );

    return {};
}
}