#include "MRPolyline2Intersect.h"
#include "MRPolyline.h"
#include "MRVector2.h"
#include "MRLine.h"
#include "MRAABBTreePolyline.h"
#include "MRInplaceStack.h"
#include "MRIntersectionPrecomputes2.h"
#include "MRRayBoxIntersection2.h"

namespace MR
{

bool isPointInsidePolyline( const Polyline2& polyline, const Vector2f& point )
{
    const auto& tree = polyline.getAABBTree();
    if ( tree.nodes().size() == 0 )
        return false;

    // we consider plusX ray here
    auto rayBoxIntersect = [] ( const Box2f& box, const Vector2f& plusXRayStart )->bool
    {
        if ( box.max.x <= plusXRayStart.x )
            return false;
        if ( box.max.y <= plusXRayStart.y )
            return false;
        if ( box.min.y > plusXRayStart.y )
            return false;
        return true;
    };
    if ( !rayBoxIntersect( tree[tree.rootNodeId()].box, point ) )
        return false;

    InplaceStack<NoInitNodeId, 32> nodesStack;
    nodesStack.push( tree.rootNodeId() );

    int intersectionCounter = 0;
    while ( !nodesStack.empty() )
    {
        const auto& node = tree[nodesStack.top()];
        nodesStack.pop();
        if ( node.leaf() )
        {
            if ( node.box.min.x >= point.x )
                ++intersectionCounter;
            else
            {
                auto uEId = node.leafId();
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
                nodesStack.push( node.l );
            if ( rayBoxIntersect( tree[node.r].box, point ) )
                nodesStack.push( node.r );
        }
    }
    return ( intersectionCounter % 2 ) == 1;
}

template<typename T>
void rayPolylineIntersectAll_( const Polyline2& polyline, const Line2<T>& line, const PolylineIntersectionCallback2<T>& callback,
    T rayStart, T rayEnd, const IntersectionPrecomputes2<T>& prec )
{
    if ( !callback )
    {
        assert( false );
        return;
    }

    const auto& tree = polyline.getAABBTree();
    if ( tree.nodes().empty() )
        return;

    // we `insignificantlyExpand` boxes to avoid leaks due to float errors
    // (small intersection of neighbor boxes guarantee that both of them will be considered as candidates of connection area)

    auto rayExpBoxIntersect = [] ( const auto& box, const auto& point, auto& t0, auto& t1, const auto& rayPrec )
    {
        return rayBoxIntersect( box.insignificantlyExpanded(), point, t0, t1, rayPrec );
    };

    T s = rayStart, e = rayEnd;
    if( !rayExpBoxIntersect( Box2<T>{ tree[tree.rootNodeId()].box }, line.p, s, e, prec ) )
        return;

    constexpr int maxTreeDepth = 32;
    std::pair< NodeId,T> nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = { tree.rootNodeId(), rayStart };

    while( currentNode >= 0 )
    {
        if( currentNode >= maxTreeDepth ) // max depth exceeded
        {
            assert( false );
            break;
        }

        const auto& node = tree[nodesStack[currentNode].first];
        if( nodesStack[currentNode--].second < rayEnd )
        {
            if( node.leaf() )
            {
                EdgeId edge = node.leafId();
                auto segm = polyline.edgeSegment( edge );
                T segmPos = 0, rayPos = 0;
                if ( doSegmentLineIntersect( LineSegm2<T>{ segm }, line, &segmPos, &rayPos )
                    && rayPos < rayEnd && rayPos > rayStart )
                {
                    if ( callback( EdgePoint{ edge, float( segmPos ) }, rayPos, rayEnd ) == Processing::Stop )
                        return;
                }
            }
            else
            {
                T lStart = rayStart, lEnd = rayEnd;
                T rStart = rayStart, rEnd = rayEnd;
                if( rayExpBoxIntersect( Box2<T>{ tree[node.l].box }, line.p, lStart, lEnd, prec ) )
                {
                    if( rayExpBoxIntersect( Box2<T>{ tree[node.r].box }, line.p, rStart, rEnd, prec ) )
                    {
                        if( lStart > rStart )
                        {
                            nodesStack[++currentNode] = { node.l,lStart };
                            nodesStack[++currentNode] = { node.r,rStart };
                        }
                        else
                        {
                            nodesStack[++currentNode] = { node.r,rStart };
                            nodesStack[++currentNode] = { node.l,lStart };
                        }
                    }
                    else
                    {
                        nodesStack[++currentNode] = { node.l,lStart };
                    }
                }
                else
                {
                    if( rayExpBoxIntersect( Box2<T>{ tree[node.r].box }, line.p, rStart, rEnd, prec ) )
                    {
                        nodesStack[++currentNode] = { node.r,rStart };
                    }
                }
            }
        }
    }
}

void rayPolylineIntersectAll( const Polyline2& polyline, const Line2f& line, const PolylineIntersectionCallback2f& callback,
    float rayStart, float rayEnd, const IntersectionPrecomputes2<float>* prec )
{
    if( prec )
    {
        return rayPolylineIntersectAll_<float>( polyline, line, callback, rayStart, rayEnd, *prec );
    }
    else
    {
        const IntersectionPrecomputes2<float> precNew( line.d );
        return rayPolylineIntersectAll_<float>( polyline, line, callback, rayStart, rayEnd, precNew );
    }
}

void rayPolylineIntersectAll( const Polyline2& polyline, const Line2d& line, const PolylineIntersectionCallback2d& callback,
    double rayStart, double rayEnd, const IntersectionPrecomputes2<double>* prec )
{
    if( prec )
    {
        return rayPolylineIntersectAll_<double>( polyline, line, callback, rayStart, rayEnd, *prec );
    }
    else
    {
        const IntersectionPrecomputes2<double> precNew( line.d );
        return rayPolylineIntersectAll_<double>( polyline, line, callback, rayStart, rayEnd, precNew );
    }
}

template<typename T>
std::optional<PolylineIntersectionResult2> rayPolylineIntersect_( const Polyline2& polyline, const Line2<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes2<T>& prec, bool closestIntersect )
{
    std::optional<PolylineIntersectionResult2> res;
    rayPolylineIntersectAll_<T>( polyline, line, [&res, closestIntersect]( const EdgePoint & polylinePoint, T rayPos, T & currRayEnd )
    {
        res = { .edgePoint = polylinePoint, .distanceAlongLine = float( rayPos ) };
        currRayEnd = rayPos;
        // stop searching if any intersection is ok
        return closestIntersect ? Processing::Continue : Processing::Stop;
    }, rayStart, rayEnd, prec );
    return res;
}

std::optional<PolylineIntersectionResult2> rayPolylineIntersect( const Polyline2& polyline, const Line2f& line,
    float rayStart, float rayEnd, const IntersectionPrecomputes2<float>* prec, bool closestIntersect )
{
    if( prec )
    {
        return rayPolylineIntersect_<float>( polyline, line, rayStart, rayEnd, *prec, closestIntersect );
    }
    else
    {
        const IntersectionPrecomputes2<float> precNew( line.d );
        return rayPolylineIntersect_<float>( polyline, line, rayStart, rayEnd, precNew, closestIntersect );
    }
}

std::optional<PolylineIntersectionResult2> rayPolylineIntersect( const Polyline2& polyline, const Line2d& line,
    double rayStart, double rayEnd, const IntersectionPrecomputes2<double>* prec, bool closestIntersect )
{
    if( prec )
    {
        return rayPolylineIntersect_<double>( polyline, line, rayStart, rayEnd, *prec, closestIntersect );
    }
    else
    {
        const IntersectionPrecomputes2<double> precNew( line.d );
        return rayPolylineIntersect_<double>( polyline, line, rayStart, rayEnd, precNew, closestIntersect );
    }
}

} //namespace MR
