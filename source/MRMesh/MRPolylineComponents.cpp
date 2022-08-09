#include "MRPolylineComponents.h"
#include "MRPolyline.h"
#include "MRPolylineTopology.h"
#include "MRPolylineEdgeIterator.h"
#include "MRTimer.h"
#include "MRVector2.h"
#include "MRVector3.h"

namespace
{

using namespace MR;

std::pair<std::vector<int>, int> getUniqueRoots( const Vector<UndirectedEdgeId, UndirectedEdgeId>& allRoots, const IteratorRange<PolylineUndirectedEdgeIterator>& region )
{
    constexpr int InvalidRoot = -1;
    std::vector<int> uniqueRootsMap( allRoots.size(), InvalidRoot );
    int k = 0;
    int curRoot;
    for ( auto e : region )
    {
        curRoot = allRoots[e];
        auto& uniqIndex = uniqueRootsMap[curRoot];
        if ( uniqIndex == InvalidRoot )
        {
            uniqIndex = k;
            ++k;
        }
    }
    return { std::move( uniqueRootsMap ), k };
}

}

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

template <typename V>
UndirectedEdgeBitSet getLargestComponent( const Polyline<V>& polyline )
{
    MR_TIMER;

    auto& topology = polyline.topology;
    auto unionFindStruct = getUnionFindStructure( topology );
    auto region = undirectedEdges( topology );

    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRoots( allRoots, region );

    auto maxLength = std::numeric_limits<float>::lowest();
    int maxI = 0;
    std::vector<float> lengths( k, 0.f );
    for ( auto e : region )
    {
        auto index = uniqueRootsMap[allRoots[e]];
        auto& length = lengths[index];
        length += polyline.edgeLength( EdgeId( e ) );
        if ( length > maxLength )
        {
            maxI = index;
            maxLength = length;
        }
    }

    UndirectedEdgeBitSet maxLengthComponent( topology.lastNotLoneEdge() + 1 );
    for ( auto e : region )
    {
        auto index = uniqueRootsMap[allRoots[e]];
        if ( index != maxI )
            continue;
        maxLengthComponent.set( e );
    }
    return maxLengthComponent;
}

template MRMESH_API UndirectedEdgeBitSet getLargestComponent<Vector2f>( const Polyline2& polyline );
template MRMESH_API UndirectedEdgeBitSet getLargestComponent<Vector3f>( const Polyline3& polyline );

}

}