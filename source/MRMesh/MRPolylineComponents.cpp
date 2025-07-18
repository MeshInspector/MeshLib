#include "MRPolylineComponents.h"
#include "MRPolyline.h"
#include "MRPolylineTopology.h"
#include "MRPolylineEdgeIterator.h"
#include "MRTimer.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRPch/MRTBB.h"
#include <climits>

namespace
{

using namespace MR;

/// returns
/// 1. the mapping: FaceId -> Root ID in [0, 1, 2, ...)
/// 2. the total number of roots
static std::pair<UndirectedEdge2RegionMap, int> getUniqueRootIds( const UndirectedEdgeMap& allRoots, const UndirectedEdgeBitSet& region )
{
    MR_TIMER;
    UndirectedEdge2RegionMap uniqueRootsMap( allRoots.size() );
    int k = 0;
    for ( auto ue : region )
    {
        auto& uniqIndex = uniqueRootsMap[allRoots[ue]];
        if ( uniqIndex < 0 )
        {
            uniqIndex = RegionId( k );
            ++k;
        }
        uniqueRootsMap[ue] = uniqIndex;
    }
    return { std::move( uniqueRootsMap ), k };
}

}

namespace MR
{

namespace PolylineComponents
{

size_t getNumComponents( const PolylineTopology& topology )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructure( topology );

    std::atomic<size_t> res{ 0 };
    tbb::parallel_for( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId( unionFindStruct.size() ) ),
        [&] ( const tbb::blocked_range<UndirectedEdgeId>& range )
    {
        size_t myRoots = 0;
        for ( auto ue = range.begin(); ue < range.end(); ++ue )
        {
            if ( !topology.hasEdge( ue ) )
                continue;
            if ( ue == unionFindStruct.findUpdateRange( ue, range.begin(), range.end() ) )
                ++myRoots;
        }
        res.fetch_add( myRoots, std::memory_order_relaxed );
    } );
    return res;
}

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

std::pair<std::vector<UndirectedEdgeBitSet>, int> getAllComponents( const PolylineTopology& topology, int maxComponentCount )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructure( topology );
    const auto& allRoots = unionFindStruct.roots();
    UndirectedEdgeBitSet region( topology.lastNotLoneUndirectedEdge() + 1 );
    for ( auto e : undirectedEdges( topology ) )
        region.set( e );
    auto [uniqueRootsMap, componentsCount] = getUniqueRootIds( allRoots, region );
    if ( !componentsCount )
        return { {}, 0 };
    const int componentsInGroup = maxComponentCount == INT_MAX ? 1 : ( componentsCount + maxComponentCount - 1 ) / maxComponentCount;
    if ( componentsInGroup != 1 )
        for ( RegionId& id : uniqueRootsMap )
            id = RegionId( id / componentsInGroup );
    componentsCount = ( componentsCount + componentsInGroup - 1 ) / componentsInGroup;
    std::vector<UndirectedEdgeBitSet> res( componentsCount );
    // this block is needed to limit allocations for not packed meshes
    std::vector<int> resSizes( componentsCount, 0 );
    for ( auto ue : undirectedEdges( topology ) )
    {
        int index = uniqueRootsMap[ue];
        if ( ue > resSizes[index] )
            resSizes[index] = ue;
    }
    for ( int i = 0; i < componentsCount; ++i )
        res[i].resize( resSizes[i] + 1 );
    // end of allocation block
    for ( auto ue : undirectedEdges( topology ) )
        res[uniqueRootsMap[ue]].set( ue );
    return { std::move( res ), componentsInGroup };
}

std::vector<MR::UndirectedEdgeBitSet> getAllComponents( const PolylineTopology& topology )
{
    return getAllComponents( topology, INT_MAX ).first;
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

    UndirectedEdgeBitSet region( topology.lastNotLoneUndirectedEdge() + 1 );
    for ( auto e : undirectedEdges( topology ) )
        region.set( e );

    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRootIds( allRoots, region );

    auto maxLength = std::numeric_limits<float>::lowest();
    int maxI = 0;
    std::vector<float> lengths( k, 0.f );
    for ( auto e : region )
    {
        auto index = uniqueRootsMap[e];
        auto& length = lengths[index];
        length += polyline.edgeLength( EdgeId( e ) );
        if ( length > maxLength )
        {
            maxI = index;
            maxLength = length;
        }
    }

    UndirectedEdgeBitSet maxLengthComponent( topology.lastNotLoneUndirectedEdge() + 1 );
    for ( auto e : region )
    {
        auto index = uniqueRootsMap[e];
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
