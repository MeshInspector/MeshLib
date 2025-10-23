#include "MRChangeMeshDataAction.h"
#include "MRMesh.h"
#include "MRPartialChangeMeshAction.h"
#include "MRChangeMeshAction.h"
#include "MRChangeVertsColorMapAction.h"
#include "MRChangeColoringActions.h"
#include "MRChangeSelectionAction.h"

namespace MR
{

PartialChangeMeshDataAction::PartialChangeMeshDataAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, ObjectMeshData&& newData )
{
    std::vector<std::shared_ptr<HistoryAction>> actions;
    actions.push_back( std::make_shared<PartialChangeMeshAction>( "mesh", obj, setNew, std::move( newData.mesh ) ) );

    actions.push_back( std::make_shared<ChangeMeshUVCoordsAction>( "setUVCoords", obj, std::move( newData.uvCoordinates ) ) );
    actions.push_back( std::make_shared<ChangeMeshTexturePerFaceAction>( "setTexturePerFace", obj, std::move( newData.texturePerFace ) ) );
    actions.push_back( std::make_shared<ChangeVertsColorMapAction<ObjectMesh>>( "setVertsColorMap", obj, std::move( newData.vertColors ) ) );
    actions.push_back( std::make_shared<ChangeFacesColorMapAction>( "setFacesColorMap", obj, std::move( newData.faceColors ) ) );
    actions.push_back( std::make_shared<ChangeMeshFaceSelectionAction>( "faceSelection", obj, std::move( newData.selectedFaces ) ) );
    actions.push_back( std::make_shared<ChangeMeshEdgeSelectionAction>( "edgeSelection", obj, std::move( newData.selectedEdges ) ) );
    actions.push_back( std::make_shared<ChangeMeshCreasesAction>( "creases", obj, std::move( newData.creases ) ) );

    combinedAction_ = std::make_unique<CombinedHistoryAction>( std::move( name ), actions );
}

}