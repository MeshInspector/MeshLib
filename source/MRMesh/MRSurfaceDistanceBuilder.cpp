#include "MRSurfaceDistanceBuilder.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRphmap.h"
#include <algorithm>

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
/// computes the distance from the source to c; returns false if the straight path from the source to c does not cross ab
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
    if ( !( sy2 > 0 ) )
        return false;
    const double cy = std::sqrt( cy2 );
    const double sy = -std::sqrt( sy2 );
    // the path from the source to c crosses line ab at x = sx + ( cx - sx ) * sy / ( sy - cy ), which must be within (0, e2)
    const double t = ( cx - sx ) * sy;
    const double h = sy - cy; // negative
    if ( !( sx * h + t < 0 && ( sx - e2 ) * h + t > 0 ) )
        return false;
    dc = std::sqrt( ( sqr( cx - sx ) + sqr( cy - sy ) ) / e2 );
    return true;
}

SurfaceDistanceBuilder::SurfaceDistanceBuilder( const Mesh & mesh, const VertBitSet* region )
    : mesh_( mesh ), region_{region}
{
    vertDistanceMap_.resize( mesh_.topology.lastValidVert() + 1, FLT_MAX );
    vertUpdatedTimes_.resize( mesh_.topology.lastValidVert() + 1, 0 );
    vertSource_.resize( mesh_.topology.lastValidVert() + 1 );
    vertLinear_.resize( mesh_.topology.lastValidVert() + 1 );
}

void SurfaceDistanceBuilder::addStartRegion( const VertBitSet & region, float startDistance )
{
    MR_TIMER;
    for ( auto v : region )
    {
        auto & vi = vertDistanceMap_[v];
        if ( vi > startDistance )
        {
            vi = startDistance;
            vertSource_[v] = addSource_( startDistance );
        }
    }
    for ( auto v : region )
        uniteTouchingSources_( v );
    sourceGroup_.roots();

    for ( auto v : region )
        suggestDistancesAround_( v );
}

void SurfaceDistanceBuilder::addStartVertices( const HashMap<VertId, float>& startVertices )
{
    MR_TIMER;
    for ( const auto & [v, dist] : startVertices )
    {
        auto & vi = vertDistanceMap_[v];
        if ( vi > dist )
        {
            vi = dist;
            vertSource_[v] = addSource_( dist );
        }
    }
    for ( const auto & [v, dist] : startVertices )
        uniteTouchingSources_( v );
    sourceGroup_.roots();

    for ( const auto & [v, dist] : startVertices )
        suggestDistancesAround_( v );
}

void SurfaceDistanceBuilder::addStart( const MeshTriPoint & start )
{
    const auto pt = mesh_.triPoint( start );
    const auto source = addSource_( 0 );
    mesh_.topology.forEachVertex( start, [&]( VertId v )
    {
        suggestVertDistance_( { v, ( mesh_.points[v] - pt ).length() }, source, false );
    } );
}

void SurfaceDistanceBuilder::setMaxVertUpdates( int v )
{
    assert( v >= 1 && v <= 255 );
    maxVertUpdates_ = std::clamp( v, 1, 255 );
}

auto SurfaceDistanceBuilder::addSource_( float startDistance ) -> StartPointId
{
    sourceStartDistance_.push_back( startDistance );
    return sourceGroup_.add();
}

void SurfaceDistanceBuilder::uniteTouchingSources_( VertId v )
{
    const auto source = vertSource_[v];
    if ( !source )
        return;
    for ( EdgeId e : orgRing( mesh_.topology, v ) )
        if ( const auto s = vertSource_[mesh_.topology.dest( e )] )
            sourceGroup_.uniteUnbalanced( s, source );
}

bool SurfaceDistanceBuilder::suggestVertDistance_( VertDistance c, StartPointId source, bool linear )
{
    auto & vi = vertDistanceMap_[c.vert];
    if ( vi > c.distance )
    {
        vi = c.distance;
        vertSource_[c.vert] = source;
        vertLinear_.set( c.vert, linear );
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
        if ( !suggestVertDistance_( c, vertSource_[v], vertLinear_.test( v ) ) )
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
    // both distances come directly from one start point, not from the linear field near a line or a region of start points
    const bool fromPoint = source == vertSource_[b] && !vertLinear_.test( a ) && !vertLinear_.test( b );
    float vc = 0;
    if ( fromPoint )
    {
        // the field is circular around the start point
        const double start = sourceStartDistance_[source];
        double dc = 0;
        if ( !getPointSourceDistAtC( Vector3d( pa ), Vector3d( pb ), Vector3d( pc ), va - start, vb - start, dc ) )
            return;
        vc = float( start + dc );
    }
    else
    {
        if ( sourceGroup_.parent( source ) != sourceGroup_.parent( vertSource_[b] ) )
            return; // the fronts from separate start points meet here, the paths along the edges are more accurate
        float dvac = 0;
        if ( !getFieldAtC( pb - pa, pc - pa, vb - va, dvac ) )
            return;
        vc = va + dvac;
        if( vc <= va )
            vc = std::nextafter( va, FLT_MAX );
    }
    suggestVertDistance_( { c, vc }, source, !fromPoint );
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
