#include "MRObjectMeshHistory.h"
#include "MRAppendHistory.h"
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRChangeSelectionAction.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRMapEdge.h>
#include <MRMesh/MRTimer.h>

namespace MR
{

void excludeLoneEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh )
{
    MR_TIMER;
    if ( !objMesh || !objMesh->mesh() )
        return;
    const auto & topology = objMesh->mesh()->topology;

    // remove deleted edges from the selection
    auto selEdges = objMesh->getSelectedEdges();
    topology.excludeLoneEdges( selEdges );
    AppendHistory<ChangeMeshEdgeSelectionAction>( "edge selection", objMesh, std::move( selEdges ) );

    // remove deleted edges from creases
    auto creases = objMesh->creases();
    topology.excludeLoneEdges( creases );
    AppendHistory<ChangeMeshCreasesAction>( "creases", objMesh, std::move( creases ) );
}

void excludeAllEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh )
{
    MR_TIMER;
    if ( !objMesh )
        return;

    // remove all edges from the selection
    AppendHistory<ChangeMeshEdgeSelectionAction>( "edge selection", objMesh, UndirectedEdgeBitSet{} );

    // remove all edges from creases
    AppendHistory<ChangeMeshCreasesAction>( "creases", objMesh, UndirectedEdgeBitSet{} );
}

void mapEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh, const WholeEdgeMap & emap )
{
    MR_TIMER;
    if ( !objMesh )
        return;

    // update edges in the selection
    auto selEdges = mapEdges( emap, objMesh->getSelectedEdges() );
    AppendHistory<ChangeMeshEdgeSelectionAction>( "edge selection", objMesh, std::move( selEdges ) );

    // update edges in the creases
    auto creases = mapEdges( emap, objMesh->creases() );
    AppendHistory<ChangeMeshCreasesAction>( "creases", objMesh, std::move( creases ) );
}

void mapEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh, const WholeEdgeHashMap & emap )
{
    MR_TIMER;
    if ( !objMesh )
        return;

    // update edges in the selection
    auto selEdges = mapEdges( emap, objMesh->getSelectedEdges() );
    AppendHistory<ChangeMeshEdgeSelectionAction>( "edge selection", objMesh, std::move( selEdges ) );

    // update edges in the creases
    auto creases = mapEdges( emap, objMesh->creases() );
    AppendHistory<ChangeMeshCreasesAction>( "creases", objMesh, std::move( creases ) );
}

void mapEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh, const UndirectedEdgeBMap & emap )
{
    MR_TIMER;
    if ( !objMesh )
        return;

    // update edges in the selection
    auto selEdges = mapEdges( emap, objMesh->getSelectedEdges() );
    AppendHistory<ChangeMeshEdgeSelectionAction>( "edge selection", objMesh, std::move( selEdges ) );

    // update edges in the creases
    auto creases = mapEdges( emap, objMesh->creases() );
    AppendHistory<ChangeMeshCreasesAction>( "creases", objMesh, std::move( creases ) );
}

} //namespace MR
