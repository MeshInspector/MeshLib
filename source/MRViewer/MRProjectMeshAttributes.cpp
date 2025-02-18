#include "MRProjectMeshAttributes.h"

#include <MRViewer/MRAppendHistory.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRMeshAttributesToUpdate.h>
#include <MRMesh/MRProjectionMeshAttribute.h>
#include <MRMesh/MRChangeMeshAction.h>
#include <MRMesh/MRChangeVertsColorMapAction.h>
#include <MRMesh/MRChangeColoringActions.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRVector.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRColor.h>
#include <MRMesh/MRId.h>

namespace MR
{

std::optional<MeshAttributes> projectMeshAttributes(
    const ObjectMesh& objectMesh,
    const MeshPart& mp,
    const AffineXf3f* newMeshXf,
    ProgressCallback cb )
{
    const auto& newMesh = mp.mesh;
    const auto& oldVertColors = objectMesh.getVertsColorMap();
    const auto& oldUVCoords = objectMesh.getUVCoords();
    const auto& oldFaceColorMap = objectMesh.getFacesColorMap();
    const auto& oldTexturePerFace = objectMesh.getTexturePerFace();

    MeshAttributes newAttribute;
    if ( !oldUVCoords.empty() )
    {
        newAttribute.uvCoords = oldUVCoords;
        newAttribute.uvCoords.resize( size_t( newMesh.topology.lastValidVert() + 1 ) );
    }
    if ( !oldVertColors.empty() )
    {
        newAttribute.colorMap = oldVertColors;
        newAttribute.colorMap.resize( size_t( newMesh.topology.lastValidVert() + 1 ) );
    }
    if ( !oldFaceColorMap.empty() )
    {
        newAttribute.faceColors = oldFaceColorMap;
        newAttribute.faceColors.resize( size_t( newMesh.topology.lastValidFace() + 1 ) );
    }
    if ( !oldTexturePerFace.empty() )
    {
        newAttribute.texturePerFace = oldTexturePerFace;
        newAttribute.texturePerFace.resize( size_t( newMesh.topology.lastValidFace() + 1 ) );
    }

    const bool hasFaceAttribs = !oldFaceColorMap.empty() || !oldTexturePerFace.empty();
    const bool hasVertAttribs = !oldVertColors.empty() || !oldUVCoords.empty();

    auto faceFunc = [&] ( FaceId id, const MeshProjectionResult& res )
    {
        if ( !oldFaceColorMap.empty() )
            newAttribute.faceColors[id] = oldFaceColorMap[res.proj.face];
        if ( !oldTexturePerFace.empty() )
            newAttribute.texturePerFace[id] = oldTexturePerFace[res.proj.face];
    };

    AffineXf3f storeXf;
    ProjectAttributeParams params;
    auto treeWXf = objectMesh.worldXf();
    params.xfs = createProjectionTransforms( storeXf, newMeshXf, &treeWXf );

    if ( hasVertAttribs )
    {
        auto vertFunc = [&] ( VertId id, const MeshProjectionResult& res, VertId v1, VertId v2, VertId v3 )
        {
            if ( !oldVertColors.empty() )
                newAttribute.colorMap[id] = res.mtp.bary.interpolate( oldVertColors[v1], oldVertColors[v2], oldVertColors[v3] );
            if ( !oldUVCoords.empty() )
                newAttribute.uvCoords[id] = res.mtp.bary.interpolate( oldUVCoords[v1], oldUVCoords[v2], oldUVCoords[v3] );
        };

        VertBitSet vertRegion;
        MeshVertPart mvp( newMesh );
        if ( mp.region )
        {
            vertRegion = getIncidentVerts( newMesh.topology, *mp.region );
            mvp.region = &vertRegion;
        }

        params.progressCb = subprogress( cb, 0.0f, hasVertAttribs ? 0.5f : 1.0f );
        if ( !projectVertAttribute( mvp, *objectMesh.mesh(), vertFunc, params ) )
            return {};

    }

    params.progressCb = subprogress( cb, hasVertAttribs ? 0.0f : 0.5f, 1.0f );
    if ( hasFaceAttribs && !projectFaceAttribute( mp, *objectMesh.mesh(), faceFunc, params ) )
        return {};

    return newAttribute;
}

void emplaceMeshAttributes(
    std::shared_ptr<ObjectMesh> objectMesh,
    MeshAttributes&& newAttribute )
{
    if ( !newAttribute.uvCoords.empty() )
    {
        Historian<ChangeMeshUVCoordsAction> htpf( "setUVCoords", objectMesh, std::move( newAttribute.uvCoords ) );
    }
    if ( !newAttribute.texturePerFace.empty() )
    {
        Historian<ChangeMeshTexturePerFaceAction> htpf( "setTexturePerFace", objectMesh, std::move( newAttribute.texturePerFace ) );
    }
    if ( !newAttribute.colorMap.empty() )
    {
        Historian<ChangeVertsColorMapAction> htpf( "setVertsColorMap", objectMesh, std::move( newAttribute.colorMap ) );
    }
    if ( !newAttribute.faceColors.empty() )
    {
        Historian<ChangeFacesColorMapAction> htpf( "setFacesColorMap", objectMesh, std::move( newAttribute.faceColors ) );
    }
}

}
