#include "MRPolylineComponents.h"
#include "MRPolylineTopology.h"
#include "MRPolylineEdgeIterator.h"
#include "MRTimer.h"

namespace MR
{

namespace PolylineComponents
{

UndirectedEdgeBitSet getComponent( const PolylineTopology& topology, UndirectedEdgeId id )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructure( topology );

    int edgeRoot = unionFindStruct.find( id );
    const auto& allRoots = unionFindStruct.roots();
    UndirectedEdgeBitSet res;
    res.resize( allRoots.size() );
    for ( auto u : undirectedEdges( topology ) )
    {
        if ( allRoots[u] == edgeRoot )
            res.set( u );
    }
    return res;
}

std::vector<UndirectedEdgeBitSet> getAllComponents( const PolylineTopology& topology )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructure( topology );

    const auto& allRoots = unionFindStruct.roots();
    constexpr int InvalidRoot = -1;
    std::vector<int> uniqueRootsMap( allRoots.size(), InvalidRoot );
    int k = 0;
    int curRoot;
    for ( auto u : undirectedEdges( topology ) )
    {
        curRoot = allRoots[u];
        auto& uniqIndex = uniqueRootsMap[curRoot];
        if ( uniqIndex == InvalidRoot )
        {
            uniqIndex = k;
            ++k;
        }
    }
    std::vector<UndirectedEdgeBitSet> res( k, UndirectedEdgeBitSet( allRoots.size() ) );
    for ( auto u : undirectedEdges( topology ) )
    {
        curRoot = allRoots[u];
        res[uniqueRootsMap[curRoot]].set( u );
    }
    return res;

}

UnionFind<MR::UndirectedEdgeId> getUnionFindStructure( const PolylineTopology& topology )
{
    MR_TIMER;

    auto size = topology.undirectedEdgeSize();

    UnionFind<UndirectedEdgeId> unionFindStructure( size );
    for ( auto u0 : undirectedEdges( topology ) )
    {
        auto u1 = topology.next( u0 );
        auto u2 = topology.next( EdgeId( u0 ).sym() );
        if ( u1.valid() && u1.undirected() != u0 )
            unionFindStructure.unite( u0, u1.undirected() );
        if ( u2.valid() && u2.undirected() != u0 )
            unionFindStructure.unite( u0, u2.undirected() );
    }
    return unionFindStructure;
}

}

}