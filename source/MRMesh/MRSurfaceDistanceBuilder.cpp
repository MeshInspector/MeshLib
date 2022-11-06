#include "MRSurfaceDistanceBuilder.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRphmap.h"
#include "MRGTest.h"

namespace MR
{

// the maximum amount of times vertex distance can be updated
static constexpr int cMaxVertUpdates = 3;

// consider triangle 0bc, where a linear scalar field is defined in two points: v(0) = 0, v(b) = b;
// computes the field in c-point;
// returns false if field gradient enters c-point not from inside of the triangle
static bool getFieldAtC( const Vector3f & b, const Vector3f & c, float vb, float & vc )
{
    assert( vb >= 0 );
    const float lb = b.length();
    if ( lb <= vb ) // equality is reached only if path gradient is along the edge, which is considered separately
        return false; // length of e-edge is less than distance of vertex values in the field (computation error?)
    const float cos_n = vb / lb;
    // n is the unit vector of field gradient in the triangle

    const auto blen = b.normalized();
    const float cos_b0c = std::min( dot( blen, c.normalized() ), 1.0f );
    if ( cos_b0c <= cos_n )
        return false; // the direction n is passing vertex c not from inside of triangle 0bc, but from 0c side

    const float cos_0bc = std::min( dot( blen, ( c - b ).normalized() ), 1.0f );
    if ( cos_0bc >= cos_n )
        return false; // the direction n is passing vertex c not from inside of triangle 0bc, but from bc side

    vc = c.length() * ( cos_b0c * cos_n + std::sqrt( 1 - sqr( cos_b0c ) ) * std::sqrt( 1 - sqr( cos_n ) ) );
    assert( vc < FLT_MAX );
    return true;
}

SurfaceDistanceBuilder::SurfaceDistanceBuilder( const Mesh & mesh, const VertBitSet* region )
    : mesh_( mesh ), region_{region}
{
    vertDistanceMap_.resize( mesh_.topology.lastValidVert() + 1, FLT_MAX );
    vertUpdatedTimes_.resize( mesh_.topology.lastValidVert() + 1, 0 );
}

void SurfaceDistanceBuilder::addStartRegion( const VertBitSet & region, float startDistance )
{
    MR_TIMER
    for ( auto v : region )
    {
        auto & vi = vertDistanceMap_[v];
        if ( vi > startDistance )
            vi = startDistance;
    }

    for ( auto v : region )
        suggestDistancesAround_( v );
}

void SurfaceDistanceBuilder::addStartVertices( const HashMap<VertId, float>& startVertices )
{
    MR_TIMER
    for ( const auto & [v, dist] : startVertices )
    {
        auto & vi = vertDistanceMap_[v];
        if ( vi > dist )
            vi = dist;
    }

    for ( const auto & [v, dist] : startVertices )
        suggestDistancesAround_( v );
}

void SurfaceDistanceBuilder::addStart( const MeshTriPoint & start )
{
    const auto pt = mesh_.triPoint( start );
    mesh_.topology.forEachVertex( start, [&]( VertId v )
    {
        suggestVertDistance_( { v, ( mesh_.points[v] - pt ).length() } );
    } );
}

bool SurfaceDistanceBuilder::suggestVertDistance_( const VertDistance & c )
{
    auto & vi = vertDistanceMap_[c.vert];
    if ( vi > c.distance )
    {
        vi = c.distance;
        if ( region_ && !region_->test( c.vert ) )
            return false;
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
        if ( !suggestVertDistance_( c ) )
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

    const auto pa = mesh_.points[a];
    const auto pb = mesh_.points[b];
    const auto pc = mesh_.points[c];

    float dvac = 0;
    if ( !getFieldAtC( pb - pa, pc - pa, vb - va, dvac ) )
        return;

    float vc = va + dvac;
    if( vc <= va )
        vc = std::nextafter( va, FLT_MAX );
    suggestVertDistance_( { c, vc } );
}

VertId SurfaceDistanceBuilder::growOne()
{
    while ( !nextVerts_.empty() )
    {
        const auto c = nextVerts_.top();
        nextVerts_.pop();
        auto & vi = vertDistanceMap_[c.vert];
        if ( vi < c.distance )
        {
            // shorter path to the vertex was found
            continue;
        }
        assert( vi == c.distance );
        auto & numUpdated = vertUpdatedTimes_[c.vert];
        if ( numUpdated >= cMaxVertUpdates )
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

TEST(MRMesh, SurfaceDistance) 
{
    float vc = 0;
    EXPECT_FALSE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, 1, vc ) );

    EXPECT_FALSE( getFieldAtC( Vector3f{ 2, 1, 0 }, Vector3f{ 3, 3, 0 }, 1, vc ) );

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.1f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.9f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 1, 0.5f, 0 }, 1 / std::sqrt(2.0f), vc ) );
    EXPECT_NEAR( vc, 1.5f / std::sqrt(2.0f), 1e-5f );
    vc = 0;
}

} //namespace MR
