#include "MRPolylineTopology.h"
#include "MRTimer.h"
#include "MRGTest.h"

namespace MR
{

EdgeId PolylineTopology::makeEdge()
{
    assert( edges_.size() % 2 == 0 );
    EdgeId he0( int( edges_.size() ) );
    EdgeId he1( int( edges_.size() + 1 ) );

    HalfEdgeRecord d0;
    d0.next = he0;
    edges_.push_back( d0 );

    HalfEdgeRecord d1;
    d1.next = he1;
    edges_.push_back( d1 );

    return he0;
}

EdgeId PolylineTopology::makeEdge( VertId a, VertId b )
{
    const auto ea = edgeWithOrg( a );
    if ( ea && next( ea ) != ea )
        return {};
    const auto eb = edgeWithOrg( b );
    if ( eb && next( eb ) != eb )
        return {};

    const auto newe = makeEdge();

    if ( ea )
        splice( ea, newe );
    else
        setOrg( newe, a );

    if ( eb )
        splice( eb, newe.sym() );
    else
        setOrg( newe.sym(), b );

    return newe;
}

bool PolylineTopology::isLoneEdge( EdgeId a ) const
{
    assert( a.valid() );
    auto & adata = edges_[a];
    if ( adata.org.valid() || adata.next != a )
        return false;

    auto b = a.sym();
    auto & bdata = edges_[b];
    if ( bdata.org.valid() || bdata.next != b )
        return false;

    return true;
}

EdgeId PolylineTopology::lastNotLoneEdge() const
{
    assert( edges_.size() % 2 == 0 );
    for ( EdgeId i{ (int)edges_.size() - 1 }; i.valid(); ----i ) // one decrement returns sym-edge
    {
        if ( !isLoneEdge( i ) )
            return i;
    }
    return {};
}

size_t PolylineTopology::computeNotLoneUndirectedEdges() const
{
    MR_TIMER;
    size_t res = 0;
    for ( EdgeId i{ 0 }; i < (int)edges_.size(); ++++i ) // one increment returns sym-edge
        if ( !isLoneEdge( i ) )
            ++res;
    return res;
}

void PolylineTopology::deleteEdge( UndirectedEdgeId ue )
{
    assert( ue.valid() );
    const EdgeId e = ue;

    if ( next( e ) != e )
        splice( next( e ), e );
    else
        setOrg( e, {} );

    if ( next( e.sym() ) != e.sym() )
        splice( next( e.sym() ), e.sym() );
    else
        setOrg( e.sym(), {} );

    assert( isLoneEdge( e ) );
}

void PolylineTopology::deleteEdges( const UndirectedEdgeBitSet & es )
{
    MR_TIMER
    for ( auto ue : es )
        deleteEdge( ue );
}

void PolylineTopology::splice( EdgeId a, EdgeId b )
{
    assert( a.valid() && b.valid() );
    if ( a == b )
        return;

    [[maybe_unused]] bool wasSameORing = next( a ) == b;
    assert( wasSameORing == ( next( b ) == a ) );

    auto & aData = edges_[a];
    [[maybe_unused]] auto & aNextData = edges_[next( a )];
    assert( aNextData.next == a );
    auto & bData = edges_[b];
    [[maybe_unused]] auto & bNextData = edges_[next( b )];
    assert( bNextData.next == b );

    bool wasSameOriginId = aData.org == bData.org;
    assert( wasSameOriginId || !aData.org.valid() || !bData.org.valid() );

    if ( !wasSameOriginId )
    {
        if ( aData.org.valid() )
            setOrg_( b, aData.org );
        else if ( bData.org.valid() )
            setOrg_( a, bData.org );
    }

    std::swap( aData.next, bData.next );
    assert( ( &aData == &aNextData ) == ( &bData == &bNextData ) );

    if ( wasSameOriginId && bData.org.valid() )
    {
        setOrg_( b, VertId() );
        if ( aData.org )
            edgePerVertex_[aData.org] = a;
    }

    assert( ( wasSameORing && next( a ) != b && next( b ) != a ) || ( !wasSameORing && next( a ) == b && next( b ) == a ) );
}

void PolylineTopology::setOrg_( EdgeId a, VertId v )
{
    assert( a.valid() );
    for ( EdgeId i = a; ; )
    {
        edges_[i].org = v;
        i = edges_[i].next;
        if ( i == a )
            break;
    }
}

void PolylineTopology::setOrg( EdgeId a, VertId v )
{
    auto oldV = org( a );
    if ( v == oldV )
        return;
    setOrg_( a, v );
    if ( oldV.valid() )
    {
        assert( edgePerVertex_[oldV].valid() );
        edgePerVertex_[oldV] = EdgeId();
        validVerts_.reset( oldV );
        --numValidVerts_;
    }
    if ( v.valid() )
    {
        assert( !edgePerVertex_[v].valid() );
        edgePerVertex_[v] = a;
        validVerts_.set( v );
        ++numValidVerts_;
    }
}

EdgeId PolylineTopology::findEdge( VertId o, VertId d ) const
{
    assert( o.valid() && d.valid() );
    EdgeId e0 = edgeWithOrg( o );
    if ( !e0.valid() )
        return {};

    for ( EdgeId e = e0;; )
    {
        if ( dest( e ) == d )
            return e;
        e = next( e );
        if ( e == e0 )
            return {};
    }
}

int PolylineTopology::getVertDegree( VertId a ) const
{
    const auto e = edgeWithOrg( a );
    if ( !e )
        return 0;
    const auto e1 = next( e );
    if ( e == e1 )
        return 1;
    assert( e == next( e1 ) );
    return 2;
}

VertId PolylineTopology::lastValidVert() const
{
    if ( numValidVerts_ <= 0 )
        return {};
    for ( VertId i{ (int)validVerts_.size() - 1 }; i.valid(); --i )
    {
        if ( validVerts_.test( i ) )
            return i;
    }
    assert( false );
    return {};
}

VertBitSet PolylineTopology::getPathVertices( const EdgePath & path ) const
{
    VertBitSet res;
    for ( auto e : path )
    {
        res.autoResizeSet( org( e ) );
        res.autoResizeSet( dest( e ) );
    }
    return res;
}

VertId PolylineTopology::splitEdge( EdgeId e )
{
    // disconnect edge e from its origin
    EdgeId eNext = next( e );
    if ( eNext != e )
        splice( eNext, e );

    // e now becomes the second part of split edge, add first part to it
    EdgeId e0 = makeEdge();
    splice( e, e0.sym() );
    if ( eNext != e )
        splice( eNext, e0 );

    // allocate id from new vertex
    VertId newv = addVertId();
    setOrg( e, newv );
    return newv;
}

EdgeId PolylineTopology::makePolyline( const VertId * vs, size_t num )
{
    if ( !vs || num < 2 )
    {
        assert( false );
        return {};
    }

    VertId maxVertId;
    for ( size_t i = 0; i < num; ++i )
        maxVertId = std::max( maxVertId, vs[i] );
    if ( maxVertId >= (int)vertSize() )
        vertResize( maxVertId + 1 );

    const auto e0 = makeEdge();
    setOrg( e0, vs[0] );
    auto e = e0;
    for ( int j = 1; j + 1 < num; ++j )
    {
        const auto ej = makeEdge();
        splice( ej, e.sym() );
        setOrg( ej, vs[j] );
        e = ej;
    }
    if ( vs[0] == vs[num-1] )
    {
        // close
        splice( e0, e.sym() );
    }
    else
    {
        setOrg( e.sym(), vs[num-1] );
    }
    return e0;
}

void PolylineTopology::write( std::ostream & s ) const
{
    // write edges
    auto numEdges = (std::uint32_t)edges_.size();
    s.write( (const char*)&numEdges, 4 );
    s.write( (const char*)edges_.data(), edges_.size() * sizeof(HalfEdgeRecord) );

    // write verts
    auto numVerts = (std::uint32_t)edgePerVertex_.size();
    s.write( (const char*)&numVerts, 4 );
    s.write( (const char*)edgePerVertex_.data(), edgePerVertex_.size() * sizeof(EdgeId) );
}

bool PolylineTopology::read( std::istream & s )
{
    // read edges
    std::uint32_t numEdges;
    s.read( (char*)&numEdges, 4 );
    if ( !s )
        return false;

    auto posCur = s.tellg();
    s.seekg( 0, std::ios_base::end );
    const auto posEnd = s.tellg();
    s.seekg( posCur );
    if ( size_t( posEnd - posCur ) < numEdges * sizeof(HalfEdgeRecord) )
        return false; // stream is too short

    edges_.resize( numEdges );
    s.read( (char*)edges_.data(), edges_.size() * sizeof(HalfEdgeRecord) );

    // read verts
    std::uint32_t numVerts;
    s.read( (char*)&numVerts, 4 );
    if ( !s )
        return false;
    edgePerVertex_.resize( numVerts );
    validVerts_.resize( numVerts );
    s.read( (char*)edgePerVertex_.data(), edgePerVertex_.size() * sizeof(EdgeId) );

    computeValidsFromEdges();

    return s.good() && checkValidity();
}

bool PolylineTopology::isConsistentlyOriented() const
{
    MR_TIMER

    for ( EdgeId e{0}; e < edges_.size(); ++e )
    {
        auto ne = next( e );
        if ( e == ne || e.odd() == ne.sym().odd() ) 
            continue;
        return false;
    }
    return true;
}

void PolylineTopology::flip()
{
    MR_TIMER

    for ( auto & e : edgePerVertex_ )
    {
        if ( e.valid() )
            e = e.sym();
    }

    for ( EdgeId i{0}; i + 1 < edges_.size(); ++++i )
    {
        auto & r0 = edges_[i];
        auto & r1 = edges_[i + 1];
        std::swap( r0, r1 );
        r0.next = r0.next.sym();
        r1.next = r1.next.sym();
    }
}

#define CHECK(x) { assert(x); if (!(x)) return false; }

bool PolylineTopology::checkValidity() const
{
    MR_TIMER

    for ( EdgeId e{0}; e < edges_.size(); ++e )
    {
        CHECK( edges_[edges_[e].next].next == e );
        if ( auto v = edges_[e].org )
            CHECK( validVerts_.test( v ) );
    }

    const auto vSize = edgePerVertex_.size();
    CHECK( vSize == validVerts_.size() )

    int realValidVerts = 0;
    for ( VertId v{0}; v < edgePerVertex_.size(); ++v )
    {
        if ( edgePerVertex_[v].valid() )
        {
            CHECK( validVerts_.test( v ) )
            const auto e0 = edgePerVertex_[v]; 
            CHECK( e0 < edges_.size() );
            CHECK( edges_[e0].org == v );
            ++realValidVerts;
            for ( EdgeId e = e0;; )
            {
                CHECK( org(e) == v );
                e = next( e );
                if ( e == e0 )
                    break;
            }
        }
        else
        {
            CHECK( !validVerts_.test( v ) )
        }
    }
    CHECK( numValidVerts_ == realValidVerts );

    return true;
}

void PolylineTopology::computeValidsFromEdges()
{
    MR_TIMER

    numValidVerts_ = 0;
    for ( VertId v{0}; v < edgePerVertex_.size(); ++v )
        if ( edgePerVertex_[v].valid() )
        {
            validVerts_.set( v );
            ++numValidVerts_;
        }
}

TEST( MRMesh, PolylineTopology )
{
    PolylineTopology t;
    VertId vs[4] = { 0_v, 1_v, 2_v, 0_v };
    t.makePolyline( vs, 4 );
    EXPECT_TRUE( t.checkValidity() );
    EXPECT_TRUE( t.isConsistentlyOriented() );
    EXPECT_EQ( t.org( 0_e ), 0_v );
    EXPECT_EQ( t.dest( 0_e ), 1_v );

    t.flip();
    EXPECT_TRUE( t.checkValidity() );
    EXPECT_TRUE( t.isConsistentlyOriented() );
    EXPECT_EQ( t.org( 0_e ), 1_v );
    EXPECT_EQ( t.dest( 0_e ), 0_v );

    EXPECT_EQ( t.numValidVerts(), 3 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 3 );

    t.deleteEdge( 0_ue );
    EXPECT_EQ( t.numValidVerts(), 3 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 2 );

    t.deleteEdge( 1_ue );
    EXPECT_EQ( t.numValidVerts(), 2 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 1 );

    t.deleteEdge( 2_ue );
    EXPECT_EQ( t.numValidVerts(), 0 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 0 );
}

} //namespace MR
