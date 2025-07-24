#include "MRMeshSubdivideCallbacks.h"

#include "MRMesh/MRColor.h"
#include "MRMesh/MRVector2.h"

namespace MR
{

OnEdgeSplit meshOnEdgeSplitAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params )
{
    OnEdgeSplit uvFunc;
    OnEdgeSplit colorFunc;
    OnEdgeSplit texturePerFaceFunc;
    OnEdgeSplit faceColorsFunc;
    if ( params.uvCoords )
        uvFunc = onEdgeSplitVertAttribute( mesh, *params.uvCoords );
    if ( params.colorMap )
        colorFunc = onEdgeSplitVertAttribute( mesh, *params.colorMap );
    if ( params.texturePerFace )
        texturePerFaceFunc = onEdgeSplitFaceAttribute( mesh, *params.texturePerFace );
    if ( params.faceColors )
        faceColorsFunc = onEdgeSplitFaceAttribute( mesh, *params.faceColors );

    auto onEdgeSplit = [&,
        uvFunc_ = std::move( uvFunc ),
        colorFunc_ = std::move( colorFunc ),
        texturePerFaceFunc_ = std::move( texturePerFaceFunc ),
        faceColorsFunc_ = std::move( faceColorsFunc )]
        (EdgeId e1, EdgeId e)
    {
        if ( params.uvCoords )
            uvFunc_( e1, e );
        if ( params.colorMap )
            colorFunc_( e1, e );
        if ( params.texturePerFace )
            texturePerFaceFunc_( e1, e );
        if ( params.faceColors )
            faceColorsFunc_( e1, e );
        return true;
    };

    return onEdgeSplit;
}

OnEdgeSplit meshOnEdgeSplitVertAttribute( const Mesh& mesh, const MeshAttributesToUpdate& params )
{
    if ( params.uvCoords && params.colorMap )
    {
        auto uvFunc = onEdgeSplitVertAttribute( mesh, *params.uvCoords );
        auto colorFunc = onEdgeSplitVertAttribute( mesh, *params.colorMap );
        auto onEdgeSplit = [&, uvFunc_ = std::move( uvFunc ), colorFunc_ = std::move( colorFunc ) ] ( EdgeId e1, EdgeId e )
        {
            uvFunc_( e1, e );
            colorFunc_( e1, e );
            return true;
        };

        return onEdgeSplit;
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
        auto onEdgeSplit = [&, texturePerFaceFunc_ = std::move( texturePerFaceFunc ), faceColorsFunc_ = std::move( faceColorsFunc )] ( EdgeId e1, EdgeId e )
        {
            texturePerFaceFunc_( e1, e );
            faceColorsFunc_( e1, e );
            return true;
        };

        return onEdgeSplit;
    }

    if ( params.texturePerFace )
        return onEdgeSplitFaceAttribute( mesh, *params.texturePerFace );
    if ( params.faceColors )
        return onEdgeSplitFaceAttribute( mesh, *params.faceColors );

    return {};
}
}
