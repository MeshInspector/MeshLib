#include "MRPolyline2Intersect.h"
#include "MRPolyline.h"
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

template<typename T>
std::optional<PolylineIntersectionResult2> rayPolylineIntersect_( const Polyline2& polyline, const Line2<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes2<T>& prec, bool closestIntersect )
{
    constexpr int maxTreeDepth = 32;
    const auto& tree = polyline.getAABBTree();
    if( tree.nodes().empty() )
        return std::nullopt;

    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( Box3<T>{ tree[tree.rootNodeId()].box }, line.p, s, e, prec ) )
    {
        return std::nullopt;
    }

    std::pair< AABBTree::NodeId,T> nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = { tree.rootNodeId(), rayStart };

    FaceId faceId;
    TriPointf triP;
    while( currentNode >= 0 && ( closestIntersect || !faceId ) )
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
                auto edge = node.leafId();
                auto segm = a.edgeSegment( edge );
                if ( doSegmentsIntersect( LineSegm<T>{ segm }, LineSegm<T>{ line( rayStart ), line( rayEnd ) } ) ) //!!!
                {
                }

                if( !meshPart.region || meshPart.region->test( face ) )
                {
                    VertId a, b, c;
                    m.topology.getTriVerts( face, a, b, c );

                    const Vector3<T> vA = Vector3<T>( m.points[a] ) - line.p;
                    const Vector3<T> vB = Vector3<T>( m.points[b] ) - line.p;
                    const Vector3<T> vC = Vector3<T>( m.points[c] ) - line.p;
                    if ( auto triIsect = rayTriangleIntersect( vA, vB, vC, prec ) )
                    {
                        if ( triIsect->t < rayEnd && triIsect->t > rayStart )
                        {
                            faceId = face;
                            triP = triIsect->bary;
                            rayEnd = triIsect->t;
                        }
                    }
                }
            }
            else
            {
                T lStart = rayStart, lEnd = rayEnd;
                T rStart = rayStart, rEnd = rayEnd;
                if( rayBoxIntersect( Box3<T>{ tree[node.l].box }, line.p, lStart, lEnd, prec ) )
                {
                    if( rayBoxIntersect( Box3<T>{ tree[node.r].box }, line.p, rStart, rEnd, prec ) )
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
                    if( rayBoxIntersect( Box3<T>{ tree[node.r].box }, line.p, rStart, rEnd, prec ) )
                    {
                        nodesStack[++currentNode] = { node.r,rStart };
                    }
                }
            }
        }
    }

    if( faceId.valid() )
    {
        MeshIntersectionResult res;
        res.proj.face = faceId;
        res.proj.point = Vector3f( line.p + rayEnd * line.d );
        res.mtp = MeshTriPoint( m.topology.edgeWithLeft( faceId ), triP );
        res.distanceAlongLine = float( rayEnd );
        return res;
    }
    else
    {
        return std::nullopt;
    }
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

} //namespace MR
