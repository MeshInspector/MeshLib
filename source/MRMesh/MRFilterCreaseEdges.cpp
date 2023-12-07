#include "MRFilterCreaseEdges.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRRegionBoundary.h"
#include "MRRingIterator.h"
#include "MRPch/MRTBB.h"

namespace MR
{
void filterCreaseEdges( const Mesh& mesh, UndirectedEdgeBitSet& creaseEdges, float critLength, bool filterComponents, bool filterBranches )
{
    if ( !filterBranches && !filterComponents )
    {
        assert( false );
        return;
    }

    if ( filterComponents )
    {
        auto selectedComponents = MeshComponents::getAllComponentsUndirectedEdges( mesh, creaseEdges );

        for ( const auto& selectedComponent : selectedComponents )
        {
            const auto componentLength = parallel_deterministic_reduce( tbb::blocked_range( 0_ue, UndirectedEdgeId{ selectedComponent.size() }, 1024 ), 0.0,
           [&] ( const auto& range, double curr )
            {
                for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
                    if ( selectedComponent.test( ue ) )
                        curr += mesh.edgeLength( ue );
                return curr;
            },
           [] ( auto a, auto b )
            {
                return a + b;
            } );

            if ( componentLength < critLength )
                creaseEdges -= selectedComponent;
        }
    }

    if ( !filterBranches )
        return;

    UndirectedEdgeBitSet backupCreaseEdges( creaseEdges );

    auto incidentVertices = getIncidentVerts( mesh.topology, creaseEdges );
    std::vector<UndirectedEdgeId> branch;

    for ( auto v : incidentVertices )
    {
        std::vector<std::pair<UndirectedEdgeId, VertId>> connections;
        for ( EdgeId e : orgRing( mesh.topology, v ) )
        {
            const UndirectedEdgeId ue = e.undirected();
            if ( backupCreaseEdges.test( ue ) )
                connections.push_back( { ue, mesh.topology.dest( e ) } );
        }

        if ( connections.size() != 1 )
            continue;

        VertId prevVert = v;
        VertId nextVert = connections[0].second;
        float branchLength = 0;
        UndirectedEdgeId ueCur = connections[0].first;
        branch.clear();

        while ( true )
        {
            branch.push_back( ueCur );
            branchLength += ( mesh.points[prevVert] - mesh.points[nextVert] ).length();
            if ( branchLength > critLength )
                break;

            connections.clear();
            for ( EdgeId e : orgRing( mesh.topology, nextVert ) )
            {
                if ( backupCreaseEdges.test( e.undirected() ) )
                    connections.push_back( { e.undirected(), mesh.topology.dest( e ) } );
            }

            if ( connections.size() == 1 || connections.size() > 2 )
                break;

            prevVert = nextVert;
            if ( connections[0].first == ueCur )
            {
                ueCur = connections[1].first;
                nextVert = connections[1].second;
            }
            else
            {
                ueCur = connections[0].first;
                nextVert = connections[0].second;
            }
        }

        if ( branchLength >= critLength )
            continue;

        for ( auto ue : branch )
        {
            creaseEdges.set( ue, false );
        }
    }
}
}