#include "MRPolylineProject.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRAABBTreePolyline.h"
#include "MRLineSegm.h"
#include <algorithm>
#include <execution>

namespace MR
{

template<typename F>
PolylineProjectionResult findProjectionCore( const Vector3f& pt, const AABBTreePolyline3 & tree, float upDistLimitSq, AffineXf3f* xf, F && edgeToEndPoints )
{
    PolylineProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.nodes().empty() )
    {
        assert( false );
        return res;
    }

    struct SubTask
    {
        AABBTreePolyline3::NodeId n;
        float distSq = 0;
        SubTask() = default;
        SubTask( AABBTreePolyline3::NodeId n, float dd ) : n( n ), distSq( dd )
        {
        }
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( AABBTreePolyline3::NodeId n )
    {
        float distSq = ( transformed( tree.nodes()[n].box, xf ).getBoxClosestPointTo( pt ) - pt ).lengthSq();
        return SubTask( n, distSq );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = tree[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            const auto lineId = node.leafId();
            Vector3f a, b;
            edgeToEndPoints( lineId, a, b );
            if ( xf )
            {
                a = ( *xf )( a );
                b = ( *xf )( b );
            }
            auto proj = closestPointOnLineSegm( pt, { a, b } );

            float distSq = ( proj - pt ).lengthSq();
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.point = proj;
                res.line = lineId;
            }
            continue;
        }

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        if ( s1.distSq < s2.distSq )
            std::swap( s1, s2 );
        assert( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
    }

    return res;
}

PolylineProjectionResult findProjectionOnPolyline( const Vector3f& pt, const Polyline3& polyline, float upDistLimitSq, AffineXf3f* xf )
{
    return findProjectionCore( pt, polyline.getAABBTree(), upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = polyline.orgPnt( ue );
        b = polyline.destPnt( ue );
    } );
}

PolylineProjectionWithOffsetResult findProjectionOnPolylineWithOffset(
    const Vector3f& pt, const Polyline3& polyline, 
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, /*< offset for each edge of polyline */ 
    float upDistLimit /*= FLT_MAX*/, /*< upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point */ 
    AffineXf3f* xf /*= nullptr */ )
{
    const AABBTreePolyline3& tree = polyline.getAABBTree();

    PolylineProjectionWithOffsetResult res;
    res.dist = upDistLimit;
    if ( tree.nodes().empty() )
    {
        assert( false );
        return res;
    }

    if ( offsetPerEdge.size() < polyline.topology.undirectedEdgeSize() )
    {
        assert( false );
        return res;
    }

#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
    float maxOffset = *std::max_element( offsetPerEdge.vec_.begin(), offsetPerEdge.vec_.end() );
#else
    float maxOffset = *std::max_element( std::execution::par, offsetPerEdge.vec_.begin(), offsetPerEdge.vec_.end() );
#endif

    struct SubTask
    {
        AABBTreePolyline3::NodeId n;
        float dist = 0;
        SubTask() = default;
        SubTask( AABBTreePolyline3::NodeId n, float d ) : n( n ), dist( d )
        {}
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.dist < res.dist )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( AABBTreePolyline3::NodeId n )
    {
        float dist = ( ( transformed( tree.nodes()[n].box, xf ).getBoxClosestPointTo( pt ) - pt ).length() - maxOffset );
        return SubTask( n, dist );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = tree[s.n];
        if ( s.dist >= res.dist )
            continue;

        if ( node.leaf() )
        {
            const auto lineId = node.leafId();
            Vector3f a, b;
            a = polyline.orgPnt( lineId );
            b = polyline.destPnt( lineId );
            if ( xf )
            {
                a = ( *xf )( a );
                b = ( *xf )( b );
            }
            auto proj = closestPointOnLineSegm( pt, { a, b } );

            float dist = ( proj - pt ).length() - offsetPerEdge[lineId];
            if ( dist < res.dist )
            {
                res.dist = dist;
                res.point = proj;
                res.line = lineId;
            }
            continue;
        }

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        if ( s1.dist < s2.dist )
            std::swap( s1, s2 );
        assert( s1.dist >= s2.dist );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
    }

    return res;
}

PolylineProjectionResult findProjectionOnMeshEdges( const Vector3f& pt, const Mesh& mesh, const AABBTreePolyline3& tree, float upDistLimitSq, AffineXf3f* xf )
{
    return findProjectionCore( pt, tree, upDistLimitSq, xf, [&]( UndirectedEdgeId ue, Vector3f & a, Vector3f & b ) 
    {
        a = mesh.orgPnt( ue );
        b = mesh.destPnt( ue );
    } );
}

} //namespace MR
