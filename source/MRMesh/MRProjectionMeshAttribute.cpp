#include "MRProjectionMeshAttribute.h"
#include "MRObjectMeshData.h"
#include "MRRegionBoundary.h"
#include "MRBitSet.h"

namespace MR
{

Expected<void> projectObjectMeshData( const ObjectMeshData& oldMeshData, ObjectMeshData& newMeshData, 
    const FaceBitSet* region /*= nullptr*/, const ProjectAttributeParams& params /*= {} */ )
{
    if ( !newMeshData.mesh || !oldMeshData.mesh )
    {
        assert( false );
        return unexpected( "No mesh for projecting attributes" );
    }
    const auto& newMesh = *newMeshData.mesh;
    const auto& oldVertColors = oldMeshData.vertColors;
    const auto& oldUVCoords = oldMeshData.uvCoordinates;
    const auto& oldFaceColorMap = oldMeshData.faceColors;
    const auto& oldTexturePerFace = oldMeshData.texturePerFace;
    const auto& oldFaceSelection = oldMeshData.selectedFaces;

    if ( !oldUVCoords.empty() )
    {
        newMeshData.uvCoordinates = oldUVCoords;
        newMeshData.uvCoordinates.resize( size_t( newMesh.topology.lastValidVert() + 1 ) );
    }
    if ( !oldVertColors.empty() )
    {
        newMeshData.vertColors = oldVertColors;
        newMeshData.vertColors.resize( size_t( newMesh.topology.lastValidVert() + 1 ) );
    }
    if ( !oldFaceColorMap.empty() )
    {
        newMeshData.faceColors = oldFaceColorMap;
        newMeshData.faceColors.resize( size_t( newMesh.topology.lastValidFace() + 1 ) );
    }
    if ( !oldTexturePerFace.empty() )
    {
        newMeshData.texturePerFace = oldTexturePerFace;
        newMeshData.texturePerFace.resize( size_t( newMesh.topology.lastValidFace() + 1 ) );
    }
    bool haveFaceSelection = oldFaceSelection.any();
    if ( haveFaceSelection )
    {
        newMeshData.selectedFaces = oldFaceSelection;
        newMeshData.selectedFaces.resize( size_t( newMesh.topology.lastValidFace() + 1 ) );
    }

    const bool hasFaceAttribs = !oldFaceColorMap.empty() || !oldTexturePerFace.empty() || haveFaceSelection;
    const bool hasVertAttribs = !oldVertColors.empty() || !oldUVCoords.empty();

    auto faceFunc = [&] ( FaceId id, const MeshProjectionResult& res )
    {
        if ( !oldFaceColorMap.empty() )
            newMeshData.faceColors[id] = oldFaceColorMap[res.proj.face];
        if ( !oldTexturePerFace.empty() )
            newMeshData.texturePerFace[id] = oldTexturePerFace[res.proj.face];
        if ( haveFaceSelection )
            newMeshData.selectedFaces.set( id, oldFaceSelection.test( res.proj.face ) );
    };

    ProjectAttributeParams localParams = params;

    if ( hasVertAttribs )
    {
        auto vertFunc = [&] ( VertId id, const MeshProjectionResult& res, VertId v1, VertId v2, VertId v3 )
        {
            if ( !oldVertColors.empty() )
                newMeshData.vertColors[id] = res.mtp.bary.interpolate( oldVertColors[v1], oldVertColors[v2], oldVertColors[v3] );
            if ( !oldUVCoords.empty() )
                newMeshData.uvCoordinates[id] = res.mtp.bary.interpolate( oldUVCoords[v1], oldUVCoords[v2], oldUVCoords[v3] );
        };

        VertBitSet vertRegion;
        MeshVertPart mvp( newMesh );
        if ( region )
        {
            vertRegion = getIncidentVerts( newMesh.topology, *region );
            mvp.region = &vertRegion;
        }


        localParams.progressCb = subprogress( params.progressCb, 0.0f, hasFaceAttribs ? 0.5f : 1.0f );
        if ( !projectVertAttribute( mvp, *oldMeshData.mesh, vertFunc, localParams ) )
            return unexpectedOperationCanceled();

    }

    localParams.progressCb = subprogress( params.progressCb, hasVertAttribs ? 0.5f : 0.0f, 1.0f );
    if ( hasFaceAttribs && !projectFaceAttribute( MeshPart( newMesh, region ), *oldMeshData.mesh, faceFunc, localParams ) )
        return unexpectedOperationCanceled();

    return {};
}

}