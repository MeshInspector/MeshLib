#include "MRPolyline2Intersect.h"
#include "MRPolyline2.h"
#include "MRVector2.h"
#include "MRAABBTreePolyline.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

bool isPointInsidePolyline( const Polyline2& polyline, const Vector2f& point )
{
    constexpr int maxTreeDepth = 32;

    const auto& tree = polyline.getAABBTree();
    if ( tree.nodes().size() == 0 )
        return false;

    auto rayBoxIntersect = [] ( const Box2f& box, const Vector2f& rayStart )->bool
    {
        if ( box.max.x <= rayStart.x )
            return false;
        if ( box.max.y <= rayStart.y )
            return false;
        if ( box.min.y > rayStart.y )
            return false;
        return true;
    };
    if ( !rayBoxIntersect( tree[tree.rootNodeId()].box, point ) )
        return false;

    AABBTreePolyline2::NodeId nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = tree.rootNodeId();

    int intersectionCounter = 0;
    while ( currentNode >= 0 )
    {
        if ( currentNode >= maxTreeDepth ) // max depth exceeded
        {
            spdlog::critical( "Maximal AABBTree depth reached!" );
            assert( false );
            break;
        }

        const auto& node = tree[nodesStack[currentNode--]];
        if ( node.leaf() )
        {
            auto uEId = node.leafId();
            if ( node.box.min.x >= point.x )
                ++intersectionCounter;
            else
            {
                const auto& org = polyline.orgPnt( uEId );
                const auto& dest = polyline.destPnt( uEId );

                double yLength = ( double( dest.y ) - double( org.y ) );
                if ( yLength != 0.0f )
                {
                    double ratio = ( double( point.y ) - double( org.y ) ) / yLength;
                    float x = float( ratio * double( dest.x ) + ( 1.0 - ratio ) * double( org.x ) );
                    if ( x >= point.x )
                        ++intersectionCounter;
                }
            }
        }
        else
        {
            if ( rayBoxIntersect( tree[node.l].box, point ) )
                nodesStack[++currentNode] = node.l;
            if ( rayBoxIntersect( tree[node.r].box, point ) )
                nodesStack[++currentNode] = node.r;
        }
    }
    return ( intersectionCounter % 2 ) == 1;
}

}