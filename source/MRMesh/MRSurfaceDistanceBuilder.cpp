#include "MRSurfaceDistanceBuilder.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRphmap.h"
#include <algorithm>

#ifdef _MSC_VER
#define MR_SD_NOINLINE __declspec( noinline )
#else
#define MR_SD_NOINLINE __attribute__( ( noinline ) )
#endif

namespace MR
{

bool getFieldAtC( const Vector3f & b, const Vector3f & c, float vb, float & vc )
{
    assert( vb >= 0 );
    const float dot_bc = dot( b, c );
    if ( dot_bc <= 0 )
        return false; // the direction n is passing vertex c not from inside of triangle 0bc, but from 0c side

    const float blenSq = b.lengthSq();
    const float vbSq = sqr( vb );
    if ( blenSq <= vbSq ) // equality is reached only if path gradient is along the edge, which is considered separately
        return false; // length of e-edge is less than distance of vertex values in the field (computation error?)
    const float sqr_cos_n = vbSq / blenSq;
    // n is the unit vector of field gradient in the triangle

    const float clenSq = c.lengthSq();
    const float bcSq = blenSq * clenSq;
    if ( bcSq <= 0 )
        return false; // b and c are too small, so their product can become zero
    float sqr_cos_b0c = sqr( dot_bc ) / bcSq;
    if ( sqr_cos_b0c <= sqr_cos_n )
        return false; // the direction n is passing vertex c not from inside of triangle 0bc, but from 0c side

    const auto a = c - b;
    const auto dot_ba = dot( b, a );
    if ( dot_ba >= 0 )
    {
        const float alenSq = a.lengthSq();
        //const float sqr_cos_0bc = sqr( dot_ba ) / ( blenSq * alenSq );
        if ( sqr( dot_ba ) >= sqr_cos_n * blenSq * alenSq )
            return false; // the direction n is passing vertex c not from inside of triangle 0bc, but from bc side
    }

    sqr_cos_b0c = std::min( sqr_cos_b0c, 1.0f );
    vc = std::sqrt( clenSq ) * ( std::sqrt( sqr_cos_b0c * sqr_cos_n ) + std::sqrt( ( 1 - sqr_cos_b0c ) * ( 1 - sqr_cos_n ) ) );
    assert( vc < FLT_MAX );
    return true;
}

/// consider triangle abc and a point source on the other side of line ab from c, at distances da from a and db from b;
/// computes the distance from the source to c; if checkCross, returns false if the straight path from the source to c does not cross ab
template <bool checkCross = true>
static bool getPointSourceDistAtC( const Vector3d & a, const Vector3d & b, const Vector3d & c, double da, double db, double & dc )
{
    const auto ab = b - a;
    const auto ac = c - a;
    const double e2 = ab.lengthSq();
    // coordinates in the triangle's plane multiplied on e = |ab|: a = (0,0), b = (e2,0), c = (cx,cy), source = (sx,sy)
    const double cx = dot( ac, ab );
    const double cy2 = e2 * ac.lengthSq() - sqr( cx );
    if ( !( e2 > 0 && cy2 > 0 ) )
        return false;
    const double sx = ( ( da - db ) * ( da + db ) + e2 ) / 2;
    const double sy2 = sqr( da ) * e2 - sqr( sx );
    if constexpr ( checkCross )
    {
        if ( !( sy2 > 0 ) )
            return false;
    }
    else if ( !( sy2 > -1e-6 * sqr( e2 ) ) )
        return false; // without the crossing check, the source on line ab (e.g. in vertex a or b) is accepted
    const double cy = std::sqrt( cy2 );
    const double sy = checkCross ? -std::sqrt( sy2 ) : -std::sqrt( std::max( sy2, 0.0 ) );
    // the path from the source to c crosses line ab at x = sx + ( cx - sx ) * sy / ( sy - cy ), which must be within (0, e2)
    const double t = ( cx - sx ) * sy;
    const double h = sy - cy; // negative
    if constexpr ( checkCross )
    {
        if ( !( sx * h + t < 0 && ( sx - e2 ) * h + t > 0 ) )
            return false;
    }
    dc = std::sqrt( ( sqr( cx - sx ) + sqr( cy - sy ) ) / e2 );
    return true;
}

SurfaceDistanceBuilder::SurfaceDistanceBuilder( const Mesh & mesh, const VertBitSet* region )
    : mesh_( mesh ), region_{region}
{
    vertDistanceMap_.resize( mesh_.topology.lastValidVert() + 1, FLT_MAX );
    vertUpdatedTimes_.resize( mesh_.topology.lastValidVert() + 1, 0 );
    vertSource_.resize( mesh_.topology.lastValidVert() + 1 );
    vertLineEnd_.resize( mesh_.topology.lastValidVert() + 1 );
}

void SurfaceDistanceBuilder::addStartRegion( const VertBitSet & region, float startDistance )
{
    MR_TIMER;
    for ( auto v : region )
        addStartVert_( v, startDistance );
    for ( auto v : region )
        uniteTouchingSources_( v );
    finishSourceGroups_();

    for ( auto v : region )
        suggestDistancesAround_( v );
}

void SurfaceDistanceBuilder::addStartVertices( const HashMap<VertId, float>& startVertices )
{
    MR_TIMER;
    for ( const auto & [v, dist] : startVertices )
        addStartVert_( v, dist );
    for ( const auto & [v, dist] : startVertices )
        uniteTouchingSources_( v );
    finishSourceGroups_();

    for ( const auto & [v, dist] : startVertices )
        suggestDistancesAround_( v );
}

void SurfaceDistanceBuilder::addStartEdgesAndPoints( const UndirectedEdgeBitSet & startEdges, const VertBitSet & startPoints, float startDistance )
{
    MR_TIMER;
    VertBitSet verts = startPoints;
    verts.resize( vertDistanceMap_.size() );
    for ( auto ue : startEdges )
    {
        verts.set( mesh_.topology.org( ue ) );
        verts.set( mesh_.topology.dest( ue ) );
    }
    for ( auto v : verts )
        addStartVert_( v, startDistance );
    for ( auto ue : startEdges )
        addStartEdge_( ue );
    finishSourceGroups_();

    for ( auto v : verts )
        suggestDistancesAround_( v );
}

void SurfaceDistanceBuilder::addStart( const MeshTriPoint & start )
{
    const auto pt = mesh_.triPoint( start );
    const auto source = addSource_( 0, {} );
    mesh_.topology.forEachVertex( start, [&]( VertId v )
    {
        suggestVertDistance_( { v, ( mesh_.points[v] - pt ).length() }, source, {} );
    } );
}

void SurfaceDistanceBuilder::setMaxVertUpdates( int v )
{
    assert( v >= 1 && v <= 255 );
    maxVertUpdates_ = std::clamp( v, 1, 255 );
}

SurfaceDistanceStartId SurfaceDistanceBuilder::addSource_( float startDistance, VertId v )
{
    sourceStartDistance_.push_back( startDistance );
    sourceVert_.push_back( v );
    return sourceGroup_.add();
}

void SurfaceDistanceBuilder::addStartVert_( VertId v, float startDistance )
{
    auto & vi = vertDistanceMap_[v];
    if ( vi > startDistance )
    {
        vi = startDistance;
        vertSource_[v] = addSource_( startDistance, v );
    }
}

SurfaceDistanceStartId SurfaceDistanceBuilder::startInVert_( VertId v ) const
{
    const auto s = vertSource_[v];
    return s && sourceVert_[s] == v ? s : SurfaceDistanceStartId{};
}

void SurfaceDistanceBuilder::addStartEdge_( EdgeId e )
{
    const auto s0 = startInVert_( mesh_.topology.org( e ) );
    const auto s1 = startInVert_( mesh_.topology.dest( e ) );
    if ( !s0 || !s1 )
        return;
    startEdges_.autoResizeSet( e.undirected() );
    sourceGroup_.uniteUnbalanced( s1, s0 );
}

bool SurfaceDistanceBuilder::distToStart_( VertId v, SurfaceDistanceStartId s0, SurfaceDistanceStartId s1, double & d ) const
{
    const auto vs = vertSource_[v];
    const auto ve = vertLineEnd_[v];
    const double dv = vertDistanceMap_[v] - sourceStartDistance_[s0];
    if ( ve.valid() == s1.valid() )
    {
        // v has the distance to the same kind of start: equal if it is the same start
        d = dv;
        return ( vs == s0 && ve == s1 ) || ( vs == s1 && ve == s0 );
    }
    // a start edge and its end point: v is in the plane of the distance to the line of the edge and the offset along it
    const auto s = s1 ? vs : s0; // the common start point
    const auto r = s1 ? ( vs == s0 ? s1 : s0 ) : ( vs == s0 ? ve : vs ); // the other end of the edge
    if ( s1 ? ( vs != s0 && vs != s1 ) : ( vs != s0 && ve != s0 ) )
        return false;
    const auto vsv = sourceVert_[s];
    const auto vrv = sourceVert_[r];
    if ( !vsv || !vrv )
        return false;
    const auto ps = mesh_.points[vsv];
    const double t = dot( mesh_.points[v] - ps, ( mesh_.points[vrv] - ps ).normalized() );
    d = s1 ? std::sqrt( std::max( 0.0, sqr( dv ) - sqr( t ) ) ) : std::hypot( dv, t );
    return true;
}

void SurfaceDistanceBuilder::nearestStartEdge_( const Vector3f & p, SurfaceDistanceStartId & s0, SurfaceDistanceStartId & s1 ) const
{
    const auto v0 = sourceVert_[s0];
    const auto v1 = sourceVert_[s1];
    if ( !v0 || !v1 )
        return;
    const auto p0 = mesh_.points[v0];
    const auto d = mesh_.points[v1] - p0;
    const float t = dot( p - p0, d );
    if ( t >= 0 && t <= d.lengthSq() )
        return;

    // p is beyond the end s of edge s0-s1: it is near another start edge from s (or further along the start edges) or near point s itself
    auto s = t < 0 ? s0 : s1;
    auto other = t < 0 ? s1 : s0;
    for ( int step = 0; step < 4; ++step )
    {
        const auto vs = sourceVert_[s];
        const auto ps = mesh_.points[vs];
        SurfaceDistanceStartId next;
        for ( EdgeId e : orgRing( mesh_.topology, vs ) )
        {
            if ( !startEdges_.test( e.undirected() ) )
                continue;
            const auto w = mesh_.topology.dest( e );
            const auto r = startInVert_( w );
            if ( !r || r == other )
                continue;
            const auto dr = mesh_.points[w] - ps;
            const float tr = dot( p - ps, dr );
            if ( tr < 0 )
                continue;
            if ( tr <= dr.lengthSq() )
            {
                s0 = s;
                s1 = r;
                return;
            }
            next = r; // p is beyond r too
        }
        if ( !next )
            break;
        other = s;
        s = next;
    }
    s0 = s;
    s1 = {};
}

void SurfaceDistanceBuilder::finishSourceGroups_()
{
    sourceGroup_.roots();
    const auto n = sourceStartDistance_.size();
    TypedBitSet<SurfaceDistanceStartId> regionRoots( n );
    for ( SurfaceDistanceStartId s( 0 ); s < n; ++s )
    {
        const auto v = sourceVert_[s];
        if ( !v )
            continue;
        int numStartNeis = 0;
        for ( EdgeId e : orgRing( mesh_.topology, v ) )
            if ( startEdges_.test( e.undirected() ) && startInVert_( mesh_.topology.dest( e ) ) )
                ++numStartNeis;
        if ( numStartNeis > 3 )
            regionRoots.set( sourceGroup_.parent( s ) );
    }
    regionSources_.clear();
    regionSources_.resize( n );
    for ( SurfaceDistanceStartId s( 0 ); s < n; ++s )
        if ( regionRoots.test( sourceGroup_.parent( s ) ) )
            regionSources_.set( s );
}

void SurfaceDistanceBuilder::uniteTouchingSources_( VertId v )
{
    if ( !startInVert_( v ) )
        return;
    for ( EdgeId e : orgRing( mesh_.topology, v ) )
        addStartEdge_( e );
}

bool SurfaceDistanceBuilder::suggestVertDistance_( VertDistance c, SurfaceDistanceStartId source, SurfaceDistanceStartId lineEnd )
{
    auto & vi = vertDistanceMap_[c.vert];
    if ( vi > c.distance )
    {
        vi = c.distance;
        vertSource_[c.vert] = source;
        vertLineEnd_[c.vert] = lineEnd;
        if ( region_ && !region_->test( c.vert ) )
            return false;
        c.distance = metricToPenalty_( c.distance, c.vert );
        nextVerts_.push( c );
        return true;
    }
    return false;
}

void SurfaceDistanceBuilder::suggestDistancesAround_( VertId v )
{
    const float vDist = vertDistanceMap_[v];
    for ( EdgeId e : orgRing( mesh_.topology, v ) )
    {
        const auto dest = mesh_.topology.dest( e );
        VertDistance c;
        c.vert = dest;
        c.distance = vDist + mesh_.edgeLength( e );
        if( c.distance <= vDist )
            c.distance = std::nextafter( vDist, FLT_MAX );
        if ( !suggestVertDistance_( c, vertSource_[v], vertLineEnd_[v] ) )
        {
            // a shorter distance is known for dest
            considerLeftTriPath_( e );
            considerLeftTriPath_( e.sym() );
        }
    }
}

void SurfaceDistanceBuilder::considerLeftTriPath_( EdgeId e )
{
    if ( !mesh_.topology.left( e ) )
        return;
    VertId a, b, c;
    mesh_.topology.getLeftTriVerts( e, a, b, c );
    float va = vertDistanceMap_[a];
    float vb = vertDistanceMap_[b];
    assert( va < FLT_MAX && vb < FLT_MAX );
    if ( vb < va )
    {
        std::swap( a, b );
        std::swap( va, vb );
    }
    assert( vb >= va );
    // c is already not farther than a: only an obtuse angle at c can give less here,
    // and then c gets that value from its triangles nearer to the start
    if ( vertDistanceMap_[c] <= va )
        return;

    const auto pa = mesh_.points[a];
    const auto pb = mesh_.points[b];
    const auto pc = mesh_.points[c];

    const auto source = vertSource_[a];
    const auto sourceB = vertSource_[b];
    // both distances come directly from one start point, not from the linear field near a line or a region of start points
    const bool fromPoint = source == sourceB && !vertLineEnd_[a] && !vertLineEnd_[b];
    if ( fromPoint )
    {
        // the field is circular around the start point
        const double start = sourceStartDistance_[source];
        double dc = 0;
        if ( getPointSourceDistAtC( Vector3d( pa ), Vector3d( pb ), Vector3d( pc ), va - start, vb - start, dc ) )
            suggestVertDistance_( { c, float( start + dc ) }, source, {} );
        return;
    }

    if ( sourceGroup_.parent( source ) != sourceGroup_.parent( sourceB ) )
        considerTriBetweenGroups_( e, a, b, c, va );
    else
        considerTriNearStarts_( a, b, c, va, vb );
}

bool SurfaceDistanceBuilder::distAtC_( const Vector3f & pa, const Vector3f & pb, const Vector3f & pc, SurfaceDistanceStartId s0, SurfaceDistanceStartId s1,
    double da, double db, double & vc, bool checkCross ) const
{
    const double start = sourceStartDistance_[s0];
    if ( !s1 )
    {
        double dc = 0;
        if ( !( checkCross ? getPointSourceDistAtC<true>( Vector3d( pa ), Vector3d( pb ), Vector3d( pc ), da, db, dc )
            : getPointSourceDistAtC<false>( Vector3d( pa ), Vector3d( pb ), Vector3d( pc ), da, db, dc ) ) )
            return false;
        vc = start + dc;
        return true;
    }
    const bool swap = db < da;
    float dvac = 0;
    if ( !getFieldAtC( swap ? pa - pb : pb - pa, pc - ( swap ? pb : pa ), float( std::abs( db - da ) ), dvac ) )
        return false;
    auto e0 = s0, e1 = s1;
    nearestStartEdge_( pc, e0, e1 );
    if ( e0 != s0 || e1 != s1 )
        return false; // c is not beside this edge
    vc = start + std::min( da, db ) + dvac;
    return true;
}

double SurfaceDistanceBuilder::straightDistToStart_( const Vector3f & p, SurfaceDistanceStartId s0, SurfaceDistanceStartId s1 ) const
{
    const auto p0 = mesh_.points[sourceVert_[s0]];
    if ( !s1 )
        return ( p - p0 ).length();
    const auto d = mesh_.points[sourceVert_[s1]] - p0;
    const float t = std::clamp( dot( p - p0, d ) / std::max( d.lengthSq(), FLT_MIN ), 0.f, 1.f );
    return ( p - p0 - t * d ).length();
}

// not inlined: inlining it in considerLeftTriPath_ costs 1.5-2% even with no fronts meeting
MR_SD_NOINLINE void SurfaceDistanceBuilder::considerTriBetweenGroups_( EdgeId e, VertId a, VertId b, VertId c, float va )
{
    // the straight path from a start to c can cross ab near the seam, where b already has smaller distance from another group;
    // the distance from the start of a to b is estimated in triangle abw on the other side of ab, and vice versa
    VertId w;
    if ( mesh_.topology.right( e ) )
    {
        VertId q0, q1;
        mesh_.topology.getLeftTriVerts( e.sym(), q0, q1, w );
    }
    if ( !w || !( vertDistanceMap_[w] < FLT_MAX ) )
        return;
    float best = FLT_MAX;
    SurfaceDistanceStartId best0, best1;
    auto tryFrom = [&]( VertId x, VertId y )
    {
        const auto s0 = vertSource_[x];
        const auto s1 = vertLineEnd_[x];
        if ( !sourceVert_[s0] || regionSources_.test( s0 ) )
            return; // no straight-line bound for a start inside a triangle, and the linear field near a region
        // the field extended over the seam can be less accurate, but never less than the straight-line distance
        const double minVc = sourceStartDistance_[s0] + straightDistToStart_( mesh_.points[c], s0, s1 );
        if ( minVc >= std::min( best, vertDistanceMap_[c] ) )
            return;
        const double dx = vertDistanceMap_[x] - sourceStartDistance_[s0];
        double dw = 0, vy = 0, vc = 0;
        if ( !distToStart_( w, s0, s1, dw ) || !distAtC_( mesh_.points[x], mesh_.points[w], mesh_.points[y], s0, s1, dx, dw, vy, false ) )
            return;
        const double dy = std::max( vy - sourceStartDistance_[s0], straightDistToStart_( mesh_.points[y], s0, s1 ) );
        if ( !distAtC_( mesh_.points[x], mesh_.points[y], mesh_.points[c], s0, s1, dx, dy, vc ) )
            return;
        float v = float( std::max( vc, minVc ) );
        if ( s1 )
            v = std::max( v, std::nextafter( va, FLT_MAX ) );
        if ( v < best )
        {
            best = v;
            best0 = s0;
            best1 = s1;
        }
    };
    tryFrom( a, b );
    tryFrom( b, a );
    if ( best < FLT_MAX )
        suggestVertDistance_( { c, best }, best0, best1 );
}

void SurfaceDistanceBuilder::considerTriNearStarts_( VertId a, VertId b, VertId c, float va, float vb )
{
    const auto pa = mesh_.points[a];
    const auto pb = mesh_.points[b];
    const auto pc = mesh_.points[c];
    const auto source = vertSource_[a];
    const auto sourceB = vertSource_[b];
    const auto lineEndA = vertLineEnd_[a];
    const auto lineEndB = vertLineEnd_[b];
    auto planeWave = [&]
    {
        float dvac = 0;
        if ( !getFieldAtC( pb - pa, pc - pa, vb - va, dvac ) )
            return FLT_MAX;
        return std::max( va + dvac, std::nextafter( va, FLT_MAX ) );
    };

    if ( regionSources_.test( source ) )
    {
        if ( const auto vc = planeWave(); vc < FLT_MAX )
            suggestVertDistance_( { c, vc }, source, lineEndA ? lineEndA : lineEndB ? lineEndB : sourceB );
        return;
    }

    // a and b are near a line of start points: each of them is labeled with its nearest start point or start edge;
    // the distances of a and b to the start of each label give c the circular field around a point or the linear field near an edge
    float best = FLT_MAX;
    SurfaceDistanceStartId best0, best1;
    auto tryStart = [&]( SurfaceDistanceStartId s0, SurfaceDistanceStartId s1 )
    {
        double da = 0, db = 0;
        if ( !distToStart_( a, s0, s1, da ) || !distToStart_( b, s0, s1, db ) )
            return;
        double dc = 0;
        if ( !distAtC_( pa, pb, pc, s0, s1, da, db, dc ) )
            return;
        float vc = float( dc );
        if ( s1 )
            vc = std::max( vc, std::nextafter( va, FLT_MAX ) );
        if ( vc < best )
        {
            best = vc;
            best0 = s0;
            best1 = s1;
        }
    };
    tryStart( source, lineEndA );
    if ( sourceB != source || lineEndB != lineEndA )
        tryStart( sourceB, lineEndB );
    if ( !lineEndA && !lineEndB && sourceVert_[source] && sourceVert_[sourceB] )
        if ( const auto e = mesh_.topology.findEdge( sourceVert_[source], sourceVert_[sourceB] ); e && startEdges_.test( e.undirected() ) )
            tryStart( source, sourceB );
    if ( best < FLT_MAX )
    {
        suggestVertDistance_( { c, best }, best0, best1 );
        return;
    }

    // otherwise find the start nearest to c, and the distances of a and b to it;
    // if they are unknown, then the plane wave through a and b
    auto s0 = lineEndA ? source : lineEndB ? sourceB : source;
    auto s1 = lineEndA ? lineEndA : lineEndB ? lineEndB : sourceB;
    nearestStartEdge_( pc, s0, s1 );
    tryStart( s0, s1 );
    if ( best < FLT_MAX )
        suggestVertDistance_( { c, best }, best0, best1 );
    else if ( const auto vc = planeWave(); vc < FLT_MAX )
        suggestVertDistance_( { c, vc }, s0, s1 );
}

VertId SurfaceDistanceBuilder::growOne()
{
    while ( !nextVerts_.empty() )
    {
        const auto c = nextVerts_.top();
        nextVerts_.pop();
        auto & vi = vertDistanceMap_[c.vert];
        const auto expectedPenalty = metricToPenalty_( vi, c.vert );
        if ( expectedPenalty < c.distance )
        {
            // shorter path to the vertex was found
            continue;
        }
        assert( expectedPenalty == c.distance );
        auto & numUpdated = vertUpdatedTimes_[c.vert];
        if ( numUpdated >= maxVertUpdates_ )
        {
            // stop updating to avoid infinite loops
            continue;
        }
        ++numUpdated;
        suggestDistancesAround_( c.vert );
        return c.vert;
    }
    return VertId();
}

float SurfaceDistanceBuilder::metricToPenalty_( float metric, VertId v ) const
{
    if ( !target_ )
        return metric;
    return metric + ( mesh_.points[v] - *target_ ).length();
}

} //namespace MR
