#include "MRObjectMeshHistory.h"
#include "MRAppendHistory.h"
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRChangeSelectionAction.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTimer.h>

namespace MR
{

void excludeLoneEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh )
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

void excludeAllEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh )
{
    MR_TIMER
    if ( !objMesh )
        return;

    // remove all edges from the selection
    Historian<ChangeMeshEdgeSelectionAction> hes( "edge selection", objMesh );
    objMesh->selectEdges( {} );

    // remove all edges from creases
    Historian<ChangeMeshCreasesAction> hcr( "creases", objMesh );
    objMesh->setCreases( {} );
}

[[nodiscard]] static UndirectedEdgeBitSet getMapping( const UndirectedEdgeBitSet & src, const EdgeMap & map )
{
    UndirectedEdgeBitSet res;
    for ( EdgeId b : src )
        if ( auto mapped = map[b] )
            res.autoResizeSet( mapped.undirected() );
    return res;
}

void mapEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh, const EdgeMap & emap )
{
    MR_TIMER
    if ( !objMesh )
        return;

    // update edges in the selection
    auto selEdges = getMapping( objMesh->getSelectedEdges(), emap );
    Historian<ChangeMeshEdgeSelectionAction> hes( "edge selection", objMesh );
    objMesh->selectEdges( selEdges );

    // update edges in the creases
    auto creases = getMapping( objMesh->creases(), emap );
    Historian<ChangeMeshCreasesAction> hcr( "creases", objMesh );
    objMesh->setCreases( std::move( creases ) );
}

} //namespace MR
