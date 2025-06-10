#include "MRPolyline2Intersect.h"
#include "MRPolyline.h"
#include "MRVector2.h"
#include "MRLine.h"
#include "MRAABBTreePolyline.h"
#include "MRIntersectionPrecomputes2.h"
#include "MRRayBoxIntersection2.h"
#include "MRPrecisePredicates3.h"
#include "MRPrecisePredicates2.h"
#include "MR2to3.h"

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

    NodeId nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = tree.rootNodeId();

    int intersectionCounter = 0;
    while ( currentNode >= 0 )
    {
        if ( currentNode >= maxTreeDepth ) // max depth exceeded
        {
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

    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( Box2<T>{ tree[tree.rootNodeId()].box }, line.p, s, e, prec ) )
        return;

    constexpr int maxTreeDepth = 32;
    std::pair< NodeId,T> nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = { tree.rootNodeId(), rayStart };

    ConvertToIntVector convToInt;
    ConvertToFloatVector convToFloat;
    Vector2f dP, eP;
    std::array<PreciseVertCoords2, 4> pvc;
    if constexpr ( std::is_same_v<T, double> )
    {
        Box3f box3;
        box3.min = to3dim( tree[tree.rootNodeId()].box.min );
        box3.max = to3dim( tree[tree.rootNodeId()].box.max );
        convToInt = getToIntConverter( Box3d( box3 ) );
        convToFloat = getToFloatConverter( Box3d( box3 ) );
        dP = Vector2f( line( s ) );
        eP = Vector2f( line( e ) );
        pvc[2].pt = to2dim( convToInt( to3dim( dP ) ) );
        pvc[2].id = VertId( polyline.topology.vertSize() );
        pvc[3].pt = to2dim( convToInt( to3dim( eP ) ) );
        pvc[3].id = pvc[2].id + 1;
    }

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
                T segmPos = 0, rayPos = 0;
                if constexpr ( std::is_same_v<T, double> )
                {
                    pvc[0].id = polyline.topology.org( edge );
                    pvc[0].pt = to2dim( convToInt( to3dim( polyline.points[pvc[0].id] ) ) );
                    pvc[1].id = polyline.topology.dest( edge );
                    pvc[1].pt = to2dim( convToInt( to3dim( polyline.points[pvc[1].id] ) ) );
                    if ( doSegmentSegmentIntersect( pvc ) )
                    {
                        auto inter = to2dim( convToFloat( to3dim( findSegmentSegmentIntersectionPrecise( pvc[0].pt, pvc[1].pt, pvc[2].pt, pvc[3].pt ) ) ) );
                        rayPos = dot( Vector2d( inter ) - line.p, line.d );
                        if ( rayPos < rayEnd && rayPos > rayStart )
                        {
                            segmPos = std::clamp( dot( Vector2d( inter ) - Vector2d( polyline.orgPnt( edge ) ), Vector2d( polyline.edgeVector( edge ) ) ), 0.0, 1.0 );
                            if ( callback( EdgePoint{ edge, float( segmPos ) }, rayPos, rayEnd ) == Processing::Stop )
                                return;
                        }
                    }
                }
                else
                {
                    auto segm = polyline.edgeSegment( edge );
                    if ( doSegmentLineIntersect( LineSegm2<T>{ segm }, line, & segmPos, & rayPos )
                        && rayPos < rayEnd&& rayPos > rayStart )
                    {
                        if ( callback( EdgePoint{ edge, float( segmPos ) }, rayPos, rayEnd ) == Processing::Stop )
                            return;
                    }
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
        return closestIntersect ? Processing::Stop : Processing::Continue;
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
