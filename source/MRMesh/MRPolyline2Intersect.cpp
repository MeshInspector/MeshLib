#include "MRPolyline2Intersect.h"
#include "MRPolyline.h"
#include "MRVector2.h"
#include "MRLine.h"
#include "MRAABBTreePolyline.h"
#include "MRInplaceStack.h"
#include "MRIntersectionPrecomputes2.h"
#include "MRRayBoxIntersection2.h"
#include "MRBitSet.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <algorithm>
#include <cmath>

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

BitSet findGridPointsInsidePolyline( const Polyline2& polyline, const Vector2i& dims, const Vector2f& origin, const Vector2f& step )
{
    MR_TIMER;
    assert( step.x > 0 );
    BitSet res;
    if ( dims.x <= 0 || dims.y <= 0 )
        return res;
    res.resize( size_t( dims.x ) * dims.y );

    const auto& tree = polyline.getAABBTree();
    if ( tree.nodes().empty() )
        return res;
    const auto rootBox = tree[tree.rootNodeId()].box;

    // the largest x in [-1, dims.x) such that the point of the row with that x is to the left of the given coordinate
    // (or at the same coordinate if !strict)
    auto lastX = [&] ( float c, bool strict )
    {
        auto isLeft = [&] ( int x )
        {
            const float px = step.x * float( x ) + origin.x;
            return strict ? px < c : px <= c;
        };
        auto x = (int)std::clamp( std::floor( ( double( c ) - origin.x ) / step.x ), -1.0, double( dims.x - 1 ) );
        while ( x >= 0 && !isLeft( x ) )
            --x;
        while ( x + 1 < dims.x && isLeft( x + 1 ) )
            ++x;
        return x;
    };

    // one ray along X per row of points; each task processes 64 whole rows, which start at a multiple of 64 bits,
    // so no two tasks write in the same block of the bit set
    tbb::enumerable_thread_specific<std::vector<int>> lastsPerThread;
    ParallelFor( 0, ( dims.y + 63 ) / 64, lastsPerThread, [&] ( int chunk, std::vector<int> & lasts )
    {
        for ( int y = chunk * 64; y < std::min( dims.y, chunk * 64 + 64 ); ++y )
        {
            const float py = step.y * float( y ) + origin.y;
            if ( !( rootBox.min.y <= py && py < rootBox.max.y ) )
                continue;

            // all edges crossing the row with the same half-open rule as in isPointInsidePolyline
            lasts.clear();
            InplaceStack<NoInitNodeId, 32> nodesStack;
            nodesStack.push( tree.rootNodeId() );
            while ( !nodesStack.empty() )
            {
                const auto& node = tree[nodesStack.top()];
                nodesStack.pop();
                if ( node.leaf() )
                {
                    const auto uEId = node.leafId();
                    const auto& org = polyline.orgPnt( uEId );
                    const auto& dest = polyline.destPnt( uEId );
                    const double ratio = ( double( py ) - double( org.y ) ) / ( double( dest.y ) - double( org.y ) );
                    const float x = float( ratio * double( dest.x ) + ( 1.0 - ratio ) * double( org.x ) );
                    // isPointInsidePolyline counts this crossing for a point if the point is not to the right of x
                    // and strictly to the left of the edge's box
                    lasts.push_back( x < node.box.max.x ? lastX( x, false ) : lastX( node.box.max.x, true ) );
                }
                else
                {
                    if ( const auto& box = tree[node.l].box; box.min.y <= py && py < box.max.y )
                        nodesStack.push( node.l );
                    if ( const auto& box = tree[node.r].box; box.min.y <= py && py < box.max.y )
                        nodesStack.push( node.r );
                }
            }
            std::sort( lasts.begin(), lasts.end() );

            // a point with lasts[j-1] < x <= lasts[j] has ( n - j ) crossings counted, and it is inside if this number is odd
            const auto n = lasts.size();
            const auto rowStart = size_t( y ) * dims.x;
            for ( size_t j = 1 - n % 2; j < n; j += 2 )
            {
                const int xBeg = j == 0 ? 0 : lasts[j - 1] + 1;
                const int xEnd = lasts[j] + 1;
                if ( xBeg < xEnd )
                    res.set( rowStart + xBeg, xEnd - xBeg, true );
            }
        }
    } );
    return res;
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
                    if ( callback( EdgePoint{ edge, float( segmPos ) }, rayPos, rayStart, rayEnd ) == Processing::Stop )
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
    rayPolylineIntersectAll_<T>( polyline, line, [&res, closestIntersect]( const EdgePoint & polylinePoint, T rayPos, T & currRayStart, T & currRayEnd )
    {
        res = { .edgePoint = polylinePoint, .distanceAlongLine = float( rayPos ) };
        if ( rayPos == 0 )
        {
            currRayStart = currRayEnd = 0;
            return Processing::Stop; // intersection exactly at ray origin
        }
        if ( rayPos < 0 )
        {
            assert( currRayStart < 0 );
            currRayStart = rayPos;
            currRayEnd = std::min( currRayEnd, -rayPos );
        }
        else
        {
            assert( currRayEnd > 0 );
            currRayEnd = rayPos;
            currRayStart = std::max( currRayStart, -rayPos );
        }
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
