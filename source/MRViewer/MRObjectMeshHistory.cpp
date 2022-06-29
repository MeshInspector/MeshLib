#include "MRObjectMeshHistory.h"
#include "MRAppendHistory.h"
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRChangeSelectionAction.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTimer.h>

namespace MR
{

void clearObjectMeshWithHistory( const std::shared_ptr<ObjectMesh>& objMesh )
{
    MR_TIMER
    if ( !objMesh || !objMesh->mesh() )
        return;
    const auto & topology = objMesh->mesh()->topology;

    // remove deleted edges from the selection
    auto selEdges = objMesh->getSelectedEdges();
    topology.excludeLoneEdges( selEdges );
    Historian<ChangeMeshEdgeSelectionAction> hes( "edge selection", objMesh );
    objMesh->selectEdges( selEdges );

    // remove deleted edges from creases
    auto creases = objMesh->creases();
    topology.excludeLoneEdges( creases );
    Historian<ChangeMeshCreasesAction> hcr( "creases", objMesh );
    objMesh->setCreases( std::move( creases ) );
}

} //namespace MR
