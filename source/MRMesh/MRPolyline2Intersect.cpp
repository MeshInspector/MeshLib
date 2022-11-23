#include "MRPolyline2Intersect.h"
#include "MRPolyline.h"
#include "MRVector2.h"
#include "MRLine.h"
#include "MRAABBTreePolyline.h"
#include "MRIntersectionPrecomputes2.h"
#include "MRRayBoxIntersection2.h"
#include "MRGTest.h"
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

template<typename T>
std::optional<PolylineIntersectionResult2> rayPolylineIntersect_( const Polyline2& polyline, const Line2<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes2<T>& prec, bool closestIntersect )
{
    constexpr int maxTreeDepth = 32;
    const auto& tree = polyline.getAABBTree();
    std::optional<PolylineIntersectionResult2> res;
    if( tree.nodes().empty() )
        return res;

    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( Box2<T>{ tree[tree.rootNodeId()].box }, line.p, s, e, prec ) )
        return res;

    std::pair< AABBTreePolyline2::NodeId,T> nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = { tree.rootNodeId(), rayStart };

    while( currentNode >= 0 && ( closestIntersect || !res ) )
    {
        if( currentNode >= maxTreeDepth ) // max depth exceeded
        {
            spdlog::critical( "Maximal AABBTree depth reached!" );
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
                T segmPos = 0, linePos = 0;
                if ( doSegmentLineIntersect( LineSegm2<T>{ segm }, line, &segmPos, &linePos )
                    && linePos < rayEnd && linePos > rayStart )
                {
                    res = PolylineIntersectionResult2{
                        .edgePoint = EdgePoint{ edge, float( segmPos ) },
                        .distanceAlongLine = float( linePos )
                    };
                    rayEnd = linePos;
                }
            }
            else
            {
                T lStart = rayStart, lEnd = rayEnd;
                T rStart = rayStart, rEnd = rayEnd;
                if( rayBoxIntersect( Box2<T>{ tree[node.l].box }, line.p, lStart, lEnd, prec ) )
                {
                    if( rayBoxIntersect( Box2<T>{ tree[node.r].box }, line.p, rStart, rEnd, prec ) )
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
                    if( rayBoxIntersect( Box2<T>{ tree[node.r].box }, line.p, rStart, rEnd, prec ) )
                    {
                        nodesStack[++currentNode] = { node.r,rStart };
                    }
                }
            }
        }
    }
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

TEST( MRMesh, Polyline2RayIntersect )
{
    Vector2f as[2] = { { 0, 1 }, { 4, 5 } };
    Polyline2 polyline;
    polyline.addFromPoints( as, 2, false );

    Line2f line( { 0, 2 }, { 2, -2 } );

    auto res = rayPolylineIntersect( polyline, line );
    ASSERT_TRUE( !!res );
    ASSERT_EQ( res->edgePoint.e, 0_e );
    ASSERT_EQ( res->edgePoint.a, 1.0f / 8 );
    ASSERT_EQ( res->distanceAlongLine, 1.0f / 4 );
}

} //namespace MR
