#include "MRPlanarPath.h"
#include "MRMesh.h"
#include "MRVector2.h"
#include "MRMeshTriPoint.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <atomic>

namespace MR
{

// given triangle 0bc in 3D and line segment 0d in 2D, and |0b|=|0d|;
// finds e, such that triangle 0bc is equal to 0de;
// returns (0,0) if |0b|=0
template <typename T>
Vector2<T> unfoldOnPlane( const Vector3<T>& b, const Vector3<T>& c, const Vector2<T>& d, bool toLeftFrom0d )
{
    const auto dotBC = dot( b, c );
    const auto crsBC = cross( b, c ).length();
    const auto dd = dot( d, d );
    if ( dd <= 0 )
        return {};
    // o is a vector of same length as d and orthogonal to d
    const Vector2<T> o = toLeftFrom0d ? Vector2<T>( -d.y, d.x ) : Vector2<T>( d.y, -d.x );
    return ( dotBC * d + crsBC * o ) / dd;
}

template MRMESH_API Vector2<float> unfoldOnPlane( const Vector3<float>& b, const Vector3<float>& c, const Vector2<float>& d, bool toLeftFrom0d );
template MRMESH_API Vector2<double> unfoldOnPlane( const Vector3<double>& b, const Vector3<double>& c, const Vector2<double>& d, bool toLeftFrom0d );

// finds an edge originated from v, which is located in the same triangle as p
static EdgeId commonEdge( const MeshTopology & topology, VertId v, const MeshTriPoint & p )
{
    for ( EdgeId e : orgRing( topology, v ) )
    {
        MeshTriPoint mtp0( p );
        MeshTriPoint mtp1( e, { 0.5f, 0.0f } );
        if ( fromSameTriangle( topology, mtp0, mtp1 ) )
            return e;
    }
    return {};
}

// finds an edge originated from v, which is located in the same triangle as p, and next edge does not satisfy this property
static EdgeId lastCommonEdge( const MeshTopology & topology, VertId v, const MeshTriPoint & p )
{
    EdgeId e0 = commonEdge( topology, v, p );
    if ( !e0 )
        return {};
    for ( int i = 0; i < 2; ++i )
    {
        EdgeId e1 = topology.next( e0 );
        MeshTriPoint mtp0( p );
        MeshTriPoint mtp1( e1, { 0.5f, 0.0f } );
        if ( fromSameTriangle( topology, mtp0, mtp1 ) )
            e0 = e1;
        else
            break;
    }
    return e0;
}

// finds an edge originated from v, which is located in the same triangle as p, and prev edge does not satisfy this property
static EdgeId firstCommonEdge( const MeshTopology & topology, VertId v, const MeshTriPoint & p )
{
    EdgeId e0 = commonEdge( topology, v, p );
    if ( !e0 )
        return {};
    for ( int i = 0; i < 2; ++i )
    {
        EdgeId e1 = topology.prev( e0 );
        MeshTriPoint mtp0( p );
        MeshTriPoint mtp1( e1, { 0.5f, 0.0f } );
        if ( fromSameTriangle( topology, mtp0, mtp1 ) )
            e0 = e1;
        else
            break;
    }
    return e0;
}

// finds position x on line x*b intersected by line containing segment [c,d]
static float lineIsect( const Vector2f & b, const Vector2f & c, const Vector2f & d )
{
    const auto c1 = cross( d, c );
    const auto c2 = cross( c - b, d - b );
    const auto cc = c1 + c2;
    if ( cc == 0 )
        return 0; // degenerate case
    return c1 / cc;
}

bool reducePathViaVertex( const Mesh & mesh, const MeshTriPoint & s, VertId v, const MeshTriPoint & e, 
    std::vector<MeshEdgePoint> & outPath, std::vector<Vector2f> & tmp, std::vector<MeshEdgePoint>& cachePath )
{
    MeshTriPoint stp( s );
    MeshTriPoint etp( e );
    if ( fromSameTriangle( mesh.topology, stp, etp ) )
    {
        // line segment from s to e is the shortest path
        return true;
    }

    const auto vp = mesh.points[ v ];
    const auto sp = mesh.triPoint( s ) - vp;
    const auto ep = mesh.triPoint( e ) - vp;
    const auto dist0 = sp.length() + ep.length();
    const auto sz0 = outPath.size();
    float distOneSide = FLT_MAX;

    EdgeId e0 = lastCommonEdge( mesh.topology, v, s );
    assert( e0 );
    EdgeId e1 = firstCommonEdge( mesh.topology, v, e );
    assert( e1 );
    if ( e0 && e1 )
    {
        tmp.clear();
        cachePath.clear();
        auto dp = mesh.destPnt( e0 ) - vp;
        tmp.push_back( Vector2f( 0, dp.length() ) );
        const Vector2f s2 = unfoldOnPlane( dp, sp, tmp.back(), false );
        if ( e0 != e1 )
        {
            for ( EdgeId ei = mesh.topology.next( e0 ); ; ei = mesh.topology.next( ei ) )
            {
                auto np = mesh.destPnt( ei ) - vp;
                tmp.push_back( unfoldOnPlane( dp, np, tmp.back(), true ) );
                dp = np;
                if ( ei == e1 )
                    break;
            }
        }
        const Vector2f e2 = unfoldOnPlane( dp, ep, tmp.back(), true );
        if ( tmp.back() != Vector2f() )
        {
            // no zero-length edges were encountered
            int i = 0;
            distOneSide = 0;
            for ( EdgeId ei = e0; i < tmp.size(); ei = mesh.topology.next( ei ), ++i )
            {
                if ( !mesh.topology.left( ei ) )
                {
                    // do not allow pass via hole space
                    distOneSide = FLT_MAX;
                    break;
                }
                auto & d = tmp[i];
                const auto x = std::clamp( lineIsect( d, s2, e2 ), 0.0f, 1.0f );
                if ( x <= TriPointf::eps )
                {
                    // passing via the same vertex
                    distOneSide = FLT_MAX;
                    break;
                }
                d *= x;
                cachePath.emplace_back( ei, x );
                if ( i == 0 )
                    distOneSide = ( d - s2 ).length();
                else
                    distOneSide += ( d - tmp[i-1] ).length();
                if ( i + 1 == tmp.size() )
                    distOneSide += ( d - e2 ).length();
            }
        }
    }

    e0 = firstCommonEdge( mesh.topology, v, s );
    assert( e0 );
    e1 = lastCommonEdge( mesh.topology, v, e );
    assert( e1 );
    if ( e0 && e1 )
    {
        tmp.clear();
        auto dp = mesh.destPnt( e0 ) - vp;
        tmp.push_back( Vector2f( 0, -dp.length() ) );
        const Vector2f s2 = unfoldOnPlane( dp, sp, tmp.back(), true );
        if ( e0 != e1 )
        {
            for ( EdgeId ei = mesh.topology.prev( e0 ); ; ei = mesh.topology.prev( ei ) )
            {
                auto np = mesh.destPnt( ei ) - vp;
                tmp.push_back( unfoldOnPlane( dp, np, tmp.back(), false ) );
                dp = np;
                if ( ei == e1 )
                    break;
            }
        }
        const Vector2f e2 = unfoldOnPlane( dp, ep, tmp.back(), false );
        if ( tmp.back() != Vector2f() )
        {
            // no zero-length edges were encountered
            int i = 0;
            float dist = 0;
            for ( EdgeId ei = e0; i < tmp.size(); ei = mesh.topology.prev( ei ), ++i )
            {
                if ( !mesh.topology.left( ei ) )
                {
                    // do not allow pass via hole space
                    dist = FLT_MAX;
                    break;
                }
                auto & d = tmp[i];
                const auto x = std::clamp( lineIsect( d, s2, e2 ), 0.0f, 1.0f );
                if ( x <= TriPointf::eps )
                {
                    // passing via the same vertex
                    dist = FLT_MAX;
                    break;
                }
                d *= x;
                outPath.emplace_back( ei, x );
                if ( i == 0 )
                    dist = ( d - s2 ).length();
                else
                    dist += ( d - tmp[i-1] ).length();
                if ( i + 1 == tmp.size() )
                    dist += ( d - e2 ).length();
            }
            if ( dist < dist0 || distOneSide < dist0 )
            {
                if ( distOneSide < dist )
                {
                    outPath.resize( sz0 );
                    outPath.insert( outPath.end(), cachePath.begin(), cachePath.end() );
                }
                return true;
            }
            outPath.resize( sz0 );
        }
    }

    // in case of second side was not unfold successfully but first side was
    if ( distOneSide < dist0 )
    {
        assert( outPath.size() == sz0 );
        outPath.insert( outPath.end(), cachePath.begin(), cachePath.end() );
        return true;
    }

    // failed to reduce path and avoid passing via the vertex
    return false;
}

class PntTag;
using PntId = Id<PntTag>;

// implements algorithm from https://page.mi.fu-berlin.de/mulzer/notes/alggeo/polySP.pdf
class PathInPlanarTriangleStrip
{
public:
    void clear();
    bool empty() const { return points_.empty(); }

    void reset( const Vector2f & start, const Vector2f & edge0left, const Vector2f & edge0right );

    // consider next edge that has same right point and new left point
    void nextEdgeNewLeft( const Vector2f & pos );
    // consider next edge that has same left point and new right point
    void nextEdgeNewRight( const Vector2f & pos );
    // returns coordinates of last edge ends
    void getLastEdge( Vector2f & l, Vector2f & r );

    // receives end point of the path and reports the answer by calling edgeCrossPosition for each passed edge in reverse order
    void find( const Vector2f & end, std::function< void(float) > edgeCrossPosition );

    // positive returned value means that the line p-q-r make left turn in q
    float cross( PntId p, PntId q, PntId r ) const { return MR::cross( points_[r] - points_[q], points_[p] - points_[q] ); }

private:
    // coordinates of points
    Vector<Vector2f, PntId> points_;
    // mapping: a point -> previous point in the shortest path to the point
    Vector<PntId, PntId> previous_;
    // mapping: a point -> next point in the shortest path ( defined for funnel vertices except for apex )
    Vector<PntId, PntId> next_;
    struct Edge { PntId left, right; };
    std::vector<Edge> edges_;
    // current funnel apex
    PntId apex_;
    // next left point after funnel apex
    PntId leftAfterApex_;
    // next right point after funnel apex
    PntId rightAfterApex_;
};

void PathInPlanarTriangleStrip::clear()
{
    points_.clear();
    previous_.clear();
    next_.clear();
    apex_ = PntId{0};
    leftAfterApex_ = rightAfterApex_ = PntId{};
    edges_.clear();
}

void PathInPlanarTriangleStrip::reset( const Vector2f & start, const Vector2f & edge0left, const Vector2f & edge0right )
{
    clear();
    points_.push_back( start );
    previous_.push_back( {} );
    next_.push_back( {} );

    auto pntLeft = PntId{ points_.size() };
    points_.push_back( edge0left );
    previous_.push_back( apex_ );
    next_.push_back( {} );
    leftAfterApex_ = pntLeft;

    auto pntRight = PntId{ points_.size() };
    points_.push_back( edge0right );
    previous_.push_back( apex_ );
    next_.push_back( {} );
    rightAfterApex_ = pntRight;

    Edge e;
    e.left = pntLeft;
    e.right = pntRight;
    edges_.push_back( e );
}

void PathInPlanarTriangleStrip::nextEdgeNewLeft( const Vector2f & pos )
{
    assert( !edges_.empty() );
    PntId prev = edges_.back().left;
    PntId curr{ points_.size() };
    points_.push_back( pos );
    previous_.push_back( {} );
    next_.push_back( {} );

    Edge e;
    e.left = curr;
    e.right = edges_.back().right;
    edges_.push_back( e );

    while ( prev != apex_ )
    {
        auto beforePrev = previous_[prev];
        assert( beforePrev.valid() );
        if ( cross( beforePrev, prev, curr ) > 0 )
        {
            previous_[curr] = prev;
            next_[prev] = curr;
            break;
        }
        prev = beforePrev;
    }
    if ( prev == apex_ )
    {
        while ( rightAfterApex_ && cross( curr, apex_, rightAfterApex_ ) < 0 )
        {
            apex_ = rightAfterApex_;
            rightAfterApex_ = next_[rightAfterApex_];
        }
        leftAfterApex_ = curr;
        previous_[curr] = apex_;
    }
}

void PathInPlanarTriangleStrip::nextEdgeNewRight( const Vector2f & pos )
{
    assert( !edges_.empty() );
    PntId prev = edges_.back().right;
    PntId curr{ points_.size() };
    points_.push_back( pos );
    previous_.push_back( {} );
    next_.push_back( {} );

    Edge e;
    e.left = edges_.back().left;
    e.right = curr;
    edges_.push_back( e );

    while ( prev != apex_ )
    {
        auto beforePrev = previous_[prev];
        assert( beforePrev.valid() );
        if ( cross( beforePrev, prev, curr ) < 0 )
        {
            previous_[curr] = prev;
            next_[prev] = curr;
            break;
        }
        prev = beforePrev;
    }
    if ( prev == apex_ )
    {
        while ( leftAfterApex_ && cross( curr, apex_, leftAfterApex_ ) > 0 )
        {
            apex_ = leftAfterApex_;
            leftAfterApex_ = next_[leftAfterApex_];
        }
        rightAfterApex_ = curr;
        previous_[curr] = apex_;
    }
}

void PathInPlanarTriangleStrip::getLastEdge( Vector2f & l, Vector2f & r )
{
    l = points_[ edges_.back().left ];
    r = points_[ edges_.back().right ];
}

void PathInPlanarTriangleStrip::find( const Vector2f & end, std::function< void(float) > edgeCrossPosition )
{
    nextEdgeNewLeft( end );

    auto curr = edges_.back().left;
    auto prev = previous_[curr];
    for ( int i = (int)edges_.size() - 2; i >= 0; --i )
    {
        if ( edges_[i].left == prev )
        {
            edgeCrossPosition( 0 );
            curr = prev;
            prev = previous_[prev];
        }
        else if ( edges_[i].right == prev )
        {
            edgeCrossPosition( 1 );
            curr = prev;
            prev = previous_[prev];
        }
        else if ( edges_[i].left ==  curr )
        {
            edgeCrossPosition( 0 );
        }
        else if ( edges_[i].right == curr )
        {
            edgeCrossPosition( 1 );
        }
        else
        {
            float lc = cross( prev, edges_[i].left, curr );
            float rc = cross( prev, edges_[i].right, curr );
            if ( lc - rc  != 0 )
            {
                edgeCrossPosition( std::clamp( lc / ( lc - rc ), 0.0f, 1.0f ) );
            }
            else 
            {
                // prev-point and curr-point are on the same line with edges_[i], return something
                edgeCrossPosition( 0.5f );
            }
        }
    }
}

class TriangleStripUnfolder
{
public:
    TriangleStripUnfolder( const Mesh & mesh ) : mesh_( mesh ) { }

    void clear();
    bool empty() const { return !lastEdge_; }

    // starts new unfolding, e1 will be oriented to have start in the left triangle
    void reset( MeshTriPoint start, MeshEdgePoint & e1 );
    // the path shall cross next the edge ei.e, ei will be oriented to have previous edge in the left triangle;
    // returns false if the edge cannot be added to the strip
    bool nextEdge( MeshEdgePoint & ei );

    void find( const MeshTriPoint & end, std::function< void(float) > edgeCrossPosition );

private:
    const Mesh & mesh_;
    EdgeId lastEdge_;
    PathInPlanarTriangleStrip strip_;
};

void TriangleStripUnfolder::clear()
{
    lastEdge_ = EdgeId{};
    strip_.clear();
}

void TriangleStripUnfolder::reset( MeshTriPoint start, MeshEdgePoint & e1 )
{
    // orient e1 to have start at left
    MeshTriPoint etp{ e1 };
    if( !fromSameTriangle( mesh_.topology, start, etp ) )
        assert( false );
    assert( etp.bary.b == 0 );
    e1 = MeshEdgePoint{ etp.e, etp.bary.a };
    lastEdge_ = e1.e;

    auto op = mesh_.orgPnt( lastEdge_ );
    auto dp = mesh_.destPnt( lastEdge_ ) - op;
    auto sp = mesh_.triPoint( start ) - op;
    const Vector2f d2{ 0, mesh_.edgeLength( lastEdge_ ) };
    const Vector2f s2 = unfoldOnPlane( dp, sp, d2, true );
    strip_.reset( s2, d2, { 0, 0 } );
}

bool TriangleStripUnfolder::nextEdge( MeshEdgePoint & e2 )
{
    assert( !e2.inVertex() );

    Vector2f o2, d2;
    strip_.getLastEdge( d2, o2 );
    Vector3f v[3];
    const EdgeId pl = mesh_.topology.prev( lastEdge_ );
   // orient e2 to have last edge at left
    if ( pl == e2.e.sym() )
        e2 = e2.sym();
    if ( pl == e2.e )
    {
        mesh_.getLeftTriPoints( e2.e, v[0], v[1], v[2] );
        Vector2f x2 = o2 + unfoldOnPlane( v[2] - v[0], v[1] - v[0], d2 - o2, false );
        strip_.nextEdgeNewLeft( x2 );
        lastEdge_ = pl;
        return true;
    }

    const EdgeId nl = mesh_.topology.next( lastEdge_.sym() ).sym();
   // orient e2 to have last edge at left
    if ( nl == e2.e.sym() )
        e2 = e2.sym();
    if ( nl == e2.e )
    {
        mesh_.getLeftTriPoints( e2.e, v[0], v[1], v[2] );
        Vector2f x2 = o2 + unfoldOnPlane( v[1] - v[2], v[0] - v[2], d2 - o2, false );
        strip_.nextEdgeNewRight( x2 );
        lastEdge_ = nl;
        return true;
    }
    return false;
}

void TriangleStripUnfolder::find( const MeshTriPoint & end, std::function< void(float) > edgeCrossPosition )
{
    assert( !empty() );
    auto op = mesh_.orgPnt( lastEdge_ );
    auto dp = mesh_.destPnt( lastEdge_ ) - op;
    auto ep = mesh_.triPoint( end ) - op;

    Vector2f o2, d2;
    strip_.getLastEdge( d2, o2 );
    Vector2f e2 = o2 + unfoldOnPlane( dp, ep, d2 - o2, false );

    strip_.find( e2, edgeCrossPosition );
}

int reducePath( const Mesh & mesh, const MeshTriPoint & start, std::vector<MeshEdgePoint> & path, const MeshTriPoint & end, int maxIter )
{
    if ( maxIter <= 0 )
        return 0;
    MR_TIMER;

    // consider points on degenerate edges as points in vertices
    for ( auto & e : path )
    {
        if ( !e.inVertex() && mesh.edgeLengthSq( e.e ) <= 0 )
        {
            e.a = 0;
            assert( e.inVertex() );
        }
    }

    std::vector<MeshEdgePoint> newPath;
    newPath.reserve( path.size() );
    std::vector<MeshEdgePoint> cacheOneSideUnfold;
    std::vector<Vector2f> tmp;
    std::vector<std::pair<int,int>> vertSpans;
    std::vector<MeshEdgePoint> rpoints; // to be added next in the new path in reverse order
    tbb::enumerable_thread_specific<TriangleStripUnfolder> stripPerThread( std::cref( mesh ) );
    for ( int i = 0; i < maxIter; ++i )
    {
        std::atomic<bool> pathTopologyChanged{false};

        // try to exit from vertices and remove repeating locations
        int j = 0;
        // there are points to add in newPath
        auto hasNext = [&]()
        {
            return !rpoints.empty() || j < path.size();
        };
        // returns next point to add in newPath, if extract=false then it will be returned again next time
        auto takeNext = [&]( bool extract )
        {
            MeshEdgePoint res;
            if ( !rpoints.empty() )
            {
                res = rpoints.back();
                if ( extract )
                    rpoints.pop_back();
            }
            else
            {
                res = path[j];
                if ( extract )
                    ++j;
            }
            return res;
        };
        // remove last point from newPath and mark topology as changed
        auto newPathPopBack = [&]()
        {
            newPath.pop_back();
            pathTopologyChanged.store( true, std::memory_order_relaxed );
        };
        // add all points in newPath, trying to walk around vertices
        while ( hasNext() )
        {
            auto np = takeNext( true );
            auto v = np.inVertex( mesh.topology );
            if ( !v )
            {
                // remove last point of new path if the path does not break (and becomes only shorter)
                while ( newPath.size() >= 2 )
                {
                    auto pp = newPath[ newPath.size() - 2 ];
                    if ( pp.inVertex() || !mesh.topology.sharedFace( np.e, pp.e ) )
                        break;
                    newPathPopBack();
                }
                if ( newPath.size() == 1 && fromSameTriangle( mesh.topology, MeshTriPoint{ start }, MeshTriPoint{ np } ) )
                    newPathPopBack();
                newPath.push_back( np );
                continue;
            }
            MeshTriPoint prev = newPath.empty() ? start : MeshTriPoint{ newPath.back() };
            // skip next points if they are in the same vertex
            while ( hasNext() && takeNext( false ).inVertex( mesh.topology ) == v )
                np = takeNext( true );
            MeshTriPoint next = hasNext() ? MeshTriPoint{ takeNext( false ) } : end;
            newPath.push_back( np );
            // put points of new path around vertex v in rpoints
            if ( reducePathViaVertex( mesh, next, v, prev, rpoints, tmp, cacheOneSideUnfold ) )
                newPathPopBack();
        }
        // remove last point of new path if the path does not break (and becomes only shorter)
        while ( newPath.size() >= 2 && fromSameTriangle( mesh.topology, MeshTriPoint{ newPath[ newPath.size() - 2 ] }, MeshTriPoint{ end } ) )
            newPathPopBack();
        if ( newPath.size() == 1 && fromSameTriangle( mesh.topology, MeshTriPoint{ start }, MeshTriPoint{ end } ) )
            newPathPopBack();

        path.swap( newPath );
        newPath.clear();

        // stop processing if path topology has not changed and straightening was done at least once
        if ( i > 0 && !pathTopologyChanged.load( std::memory_order_relaxed ) )
            return i + 1;

        // find all spans where points are not in a vertex
        vertSpans.clear();
        int spanStart = -1;
        for ( j = 0; j < path.size(); ++j )
        {
            if ( !path[j].inVertex() )
                continue;
            if ( spanStart + 1 < j )
                vertSpans.emplace_back( spanStart, j );
            spanStart = j;
        }
        if ( spanStart + 1 < path.size() )
            vertSpans.emplace_back( spanStart, (int)path.size() );

        // straighten path in each triangle strip
        tbb::parallel_for( tbb::blocked_range( 0, (int)vertSpans.size(), 1 ),
            [&]( const tbb::blocked_range<int> & range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                auto span = vertSpans[i];
                auto & strip = stripPerThread.local();
                while ( span.first + 1 < span.second )
                {
                    strip.clear();
                    const MeshTriPoint first = span.first < 0 ? start : MeshTriPoint{ path[span.first] };
                    strip.reset( first, path[span.first + 1] );
                    int j = span.first + 2;
                    for ( ; j < span.second; ++j )
                    {
                        // path[j] is an ordinary point not in a vertex
                        if ( !strip.nextEdge( path[j] ) )
                            break;
                    }
                    const MeshTriPoint last = j < path.size() ? MeshTriPoint{ path[j] } : end;
                    int pos = j;
                    strip.find( last, [&]( float v ) 
                    {
                        assert( pos > 0 );
                        auto & edgePoint = path[ --pos ];
                        assert( !edgePoint.inVertex() );
                        edgePoint.a = 1 - v;
                        if ( edgePoint.inVertex() && !pathTopologyChanged.load( std::memory_order_relaxed ) )
                            pathTopologyChanged.store( true, std::memory_order_relaxed );
                    } );
                    span.first = j;
                }
            }
        } );

        // stop processing if path topology has not changed
        if ( !pathTopologyChanged.load( std::memory_order_relaxed ) )
            return i + 1;
    }
    return maxIter;
}

} //namespace MR
