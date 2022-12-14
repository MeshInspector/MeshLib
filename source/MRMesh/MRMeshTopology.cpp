#include "MRMeshTopology.h"
#include "MRRingIterator.h"
#include "MRRegionBoundary.h"
#include "MREdgeIterator.h"
#include "MREdgePaths.h"
#include "MRBuffer.h"
#include "MRphmap.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"
#include "MRPch/MRTBB.h"
#include "MRProgressReadWrite.h"

namespace MR
{

EdgeId MeshTopology::makeEdge()
{
    assert( edges_.size() % 2 == 0 );
    EdgeId he0( int( edges_.size() ) );
    EdgeId he1( int( edges_.size() + 1 ) );

    HalfEdgeRecord d0;
    d0.next = d0.prev = he0;
    edges_.push_back( d0 );

    HalfEdgeRecord d1;
    d1.next = d1.prev = he1;
    edges_.push_back( d1 );

    return he0;
}

bool MeshTopology::isLoneEdge( EdgeId a ) const
{
    assert( a.valid() );
    if ( a >= edges_.size() )
        return true;
    auto & adata = edges_[a];
    if ( adata.left.valid() || adata.org.valid() || adata.next != a || adata.prev != a )
        return false;

    auto b = a.sym();
    auto & bdata = edges_[b];
    if ( bdata.left.valid() || bdata.org.valid() || bdata.next != b || bdata.prev != b )
        return false;

    return true;
}

EdgeId MeshTopology::lastNotLoneEdge() const
{
    assert( edges_.size() % 2 == 0 );
    for ( EdgeId i{ (int)edges_.size() - 1 }; i.valid(); ----i ) // one decrement returns sym-edge
    {
        if ( !isLoneEdge( i ) )
            return i;
    }
    return {};
}

void MeshTopology::excludeLoneEdges( UndirectedEdgeBitSet & edges ) const
{
    MR_TIMER
    for ( auto ue : edges )
        if ( isLoneEdge( ue ) )
            edges.reset( ue );
}

size_t MeshTopology::computeNotLoneUndirectedEdges() const
{
    MR_TIMER

    return parallel_reduce( tbb::blocked_range( 0_ue, UndirectedEdgeId{ undirectedEdgeSize() } ), size_t(0),
    [&] ( const auto & range, size_t curr )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            if ( !isLoneEdge( ue ) )
                ++curr;
        }
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
}

size_t MeshTopology::heapBytes() const
{
    return
        edges_.heapBytes() +
        edgePerVertex_.heapBytes() +
        validVerts_.heapBytes() +
        edgePerFace_.heapBytes() +
        validFaces_.heapBytes();
}

void MeshTopology::splice( EdgeId a, EdgeId b )
{
    assert( a.valid() && b.valid() );
    if ( a == b )
        return;

    auto & aData = edges_[a];
    auto & aNextData = edges_[next( a )];
    auto & bData = edges_[b];
    auto & bNextData = edges_[next( b )];

    bool wasSameOriginId = aData.org == bData.org;
    assert( wasSameOriginId || !aData.org.valid() || !bData.org.valid() );

    bool wasSameLeftId = aData.left == bData.left;
    assert( wasSameLeftId || !aData.left.valid() || !bData.left.valid() );

    if ( !wasSameOriginId )
    {
        if ( aData.org.valid() )
            setOrg_( b, aData.org );
        else if ( bData.org.valid() )
            setOrg_( a, bData.org );
    }

    if ( !wasSameLeftId )
    {
        if ( aData.left.valid() )
            setLeft_( b, aData.left );
        else if ( bData.left.valid() )
            setLeft_( a, bData.left );
    }

    std::swap( aData.next, bData.next );
    std::swap( aNextData.prev, bNextData.prev );

    if ( wasSameOriginId && bData.org.valid() )
    {
        setOrg_( b, VertId() );
        edgePerVertex_[aData.org] = a;
    }

    if ( wasSameLeftId && bData.left.valid() )
    {
        setLeft_( b, FaceId() );
        edgePerFace_[aData.left] = a;
    }
}

bool MeshTopology::fromSameOriginRing( EdgeId a, EdgeId b ) const
{
    assert( a.valid() && b.valid() );
    EdgeId ia = a;
    EdgeId ib = b;
    // simultaneously rotate in two distinct directions to finish fast even if one of the rings is large
    for(;;)
    {
        if ( ia == ib ) return true;
        ia = next( ia );
        if ( ia == a )  return false;
        if ( ia == ib ) return true;
        ib = prev( ib );
        if ( ib == b )  return false;
    } 
}

bool MeshTopology::fromSameLeftRing( EdgeId a, EdgeId b ) const
{
    assert( a.valid() && b.valid() );
    EdgeId ia = a;
    EdgeId ib = b;
    // simultaneously rotate in two distinct directions to finish fast even if one of the rings is large
    for(;;)
    {
        if ( ia == ib ) return true;
        ia = prev( ia.sym() );
        if ( ia == a )  return false;
        if ( ia == ib ) return true;
        ib = next( ib ).sym();
        if ( ib == b )  return false;
    } 
}

int MeshTopology::getLeftDegree( EdgeId a ) const
{
    assert( a.valid() );
    EdgeId b = a;
    int degree = 0;
    do
    {
        b = prev( b.sym() );
        ++degree;
    } while ( a != b );
    return degree;
}

bool MeshTopology::isLeftTri( EdgeId a ) const
{
    assert( a.valid() );
    EdgeId b = prev( a.sym() );
    if ( a == b )
        return false;
    EdgeId c = prev( b.sym() );
    if ( a == c )
        return false;
    EdgeId d = prev( c.sym() );
    return a == d;
}

void MeshTopology::getLeftTriVerts( EdgeId a, VertId & v0, VertId & v1, VertId & v2 ) const
{
    v0 = org( a );
    EdgeId b = prev( a.sym() );
    assert( a != b );
    v1 = org( b );
    EdgeId c = prev( b.sym() );
    assert( a != c );
    v2 = org( c );
    assert( a == prev( c.sym() ) );
}

void MeshTopology::getTriEdges( FaceId f, EdgeId & e0, EdgeId & e1, EdgeId & e2 ) const
{
    e0 = edgeWithLeft( f );
    e1 = prev( e0.sym() );
    e2 = prev( e1.sym() );
    assert( e0 == prev( e2.sym() ) );
}

std::vector<ThreeVertIds> MeshTopology::getAllTriVerts() const
{
    MR_TIMER
    std::vector<ThreeVertIds> res;
    res.reserve( numValidFaces_ );
    for ( auto f : validFaces_ )
    {
        VertId vs[3];
        getTriVerts( f, vs );
        res.push_back( { vs[0], vs[1], vs[2] } );
    }

    return res;
}

bool MeshTopology::isLeftQuad( EdgeId a ) const
{
    assert( a.valid() );
    EdgeId b = prev( a.sym() );
    if ( a == b )
        return false;
    EdgeId c = prev( b.sym() );
    if ( a == c )
        return false;
    EdgeId d = prev( c.sym() );
    if ( a == d )
        return false;
    EdgeId e = prev( d.sym() );
    return a == e;
}

EdgeId MeshTopology::bdEdgeSameOrigin( EdgeId e, const FaceBitSet * region ) const
{
    for ( auto ei : orgRing( *this, e ) )
        if ( isBdEdge( ei, region ) )
            return ei;
    return {};
}

void MeshTopology::setOrg_( EdgeId a, VertId v )
{
    assert( a.valid() );
    for ( EdgeId i : orgRing( *this, a ) )
    {
        edges_[i].org = v;
    }
}

void MeshTopology::setOrg( EdgeId a, VertId v )
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

void MeshTopology::setLeft_( EdgeId a, FaceId f )
{
    assert( a.valid() );
    for ( EdgeId i : leftRing( *this, a ) )
    {
        edges_[i].left = f;
    }
}

void MeshTopology::setLeft( EdgeId a, FaceId f )
{
    auto oldF = left( a );
    if ( f == oldF )
        return;
    setLeft_( a, f );
    if ( oldF.valid() )
    {
        assert( edgePerFace_[oldF].valid() );
        edgePerFace_[oldF] = EdgeId();
        validFaces_.reset( oldF );
        --numValidFaces_;
    }
    if ( f.valid() )
    {
        assert( !edgePerFace_[f].valid() );
        edgePerFace_[f] = a;
        validFaces_.set( f );
        ++numValidFaces_;
    }
}

EdgeId MeshTopology::findEdge( VertId o, VertId d ) const
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

bool MeshTopology::isClosed() const
{
    MR_TIMER
    for ( EdgeId e(0); e < edges_.size(); ++e )
    {
        if ( !edges_[e].org.valid() )
            continue; // skip edges without valid vertices
        if ( !edges_[e].left.valid() )
            return false;
    }
    return true;
}

bool MeshTopology::isClosed( const FaceBitSet * region ) const
{
    if ( !region )
        return isClosed();

    MR_TIMER
    for ( FaceId f : *region )
    {
        for ( EdgeId e : leftRing( *this, f ) )
        {
            if ( !right( e ) )
                return false;
        }
    }
    return true;
}

EdgeLoop MeshTopology::trackBoundaryLoop( EdgeId e0, const FaceBitSet * region ) const
{
    auto res = MR::trackRegionBoundaryLoop( *this, e0.sym(), region );
    if ( res.empty() )
    {
        assert( false );
        return res;
    }
    MR::reverse( res );
    std::rotate( res.begin(), res.end() - 1, res.end() ); // put e0 in res.front()
    assert( res.front() == e0 );
    return res;
}

std::vector<EdgeLoop> MeshTopology::findBoundary( const FaceBitSet * region ) const
{
    MR_TIMER

    std::vector<EdgeLoop> res;
    phmap::flat_hash_set<EdgeId> reportedBdEdges;

    for ( EdgeId e(0); e < edges_.size(); ++e )
    {
        if ( !isLeftBdEdge( e.sym(), region ) )
            continue;
        if ( !edges_[e].org.valid() )
            continue; // skip edges without valid vertices
        if ( !reportedBdEdges.insert( e ).second )
            continue;

        auto loop = trackBoundaryLoop( e, region );
        assert( loop.front() == e );
        for ( int i = 1; i < loop.size(); ++i )
        {
            [[maybe_unused]] bool inserted = reportedBdEdges.insert( loop[i] ).second;
            assert( inserted );
        }
        res.push_back( std::move( loop ) );
    }

    return res;
}

std::vector<EdgeId> MeshTopology::findHoleRepresentiveEdges() const
{
    auto bds = findBoundary();

    std::vector<EdgeId> res;
    res.reserve( bds.size() );

    for ( const auto & bd : bds )
        res.push_back( bd.front() );

    return res;
}

EdgeLoop MeshTopology::getLeftRing( EdgeId e ) const
{
    EdgeLoop res;
    for ( auto edge : leftRing( *this, e ) )
        res.push_back( edge );
    return res;
}

std::vector<EdgeLoop> MeshTopology::getLeftRings( const std::vector<EdgeId> & es ) const
{
    MR_TIMER
    std::vector<EdgeLoop> res;
    EdgeBitSet inRes;
    for ( auto e : es )
    {
        if ( inRes.test( e ) )
            continue;
        EdgeLoop loop;
        for ( auto edge : leftRing( *this, e ) )
        {
            loop.push_back( edge );
            inRes.autoResizeSet( edge );
        }
        res.push_back( std::move( loop ) );
    }
    return res;
}

EdgeBitSet MeshTopology::findBoundaryEdges() const
{
    MR_TIMER
    EdgeBitSet res;
    const EdgeId elast = lastNotLoneEdge();
    for ( EdgeId e{0}; e <= elast; ++e )
    {
        if ( !left( e ) && right( e ) )
            res.autoResizeSet( e );
    }
    return res;
}

FaceBitSet MeshTopology::findBoundaryFaces() const
{
    MR_TIMER
    FaceBitSet res;
    const EdgeId elast = lastNotLoneEdge();
    for ( EdgeId e{0}; e <= elast; ++e )
    {
        FaceId r;
        if ( !left( e ) && ( r = right( e ) ).valid() )
            res.autoResizeSet( r );
    }
    return res;
}

VertBitSet MeshTopology::findBoundaryVerts() const
{
    MR_TIMER
    VertBitSet res;
    const EdgeId elast = lastNotLoneEdge();
    for ( EdgeId e{0}; e <= elast; ++e )
    {
        if ( !left( e ) && right( e ) )
        {
            if ( auto o = org( e ) )
                res.autoResizeSet( o );
            if ( auto d = dest( e ) )
                res.autoResizeSet( d );
        }
    }
    return res;
}

VertBitSet MeshTopology::getPathVertices( const EdgePath & path ) const
{
    VertBitSet res;
    for ( auto e : path )
    {
        res.autoResizeSet( org( e ) );
        res.autoResizeSet( dest( e ) );
    }
    return res;
}

FaceBitSet MeshTopology::getPathLeftFaces( const EdgePath & path ) const
{
    FaceBitSet res;
    for ( auto e : path )
    {
        if ( auto l = left( e ) )
            res.autoResizeSet( l );
    }
    return res;
}

FaceBitSet MeshTopology::getPathRightFaces( const EdgePath & path ) const
{
    FaceBitSet res;
    for ( auto e : path )
    {
        if ( auto r = right( e ) )
            res.autoResizeSet( r );
    }
    return res;
}

VertId MeshTopology::lastValidVert() const
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

EdgeId MeshTopology::sharedEdge( FaceId l, FaceId r ) const
{
    assert( l && r && l != r );
    for ( auto e : leftRing( *this, l ) )
        if ( right( e ) == r )
            return e;
    return {};
}

EdgeId MeshTopology::sharedVertInOrg( EdgeId a, EdgeId b ) const
{
    assert ( a && b );
    if ( a == b || a == b.sym() )
        return a;
    const auto ao = org( a );
    const auto bo = org( b );
    assert( ao && bo );
    if ( ao == bo )
        return a;
    const auto ad = dest( a );
    if ( ad == bo )
        return a.sym();
    const auto bd = dest( b );
    if ( ao == bd )
        return a;
    if ( ad == bd )
        return a.sym();
    return {}; // no shared vertex found
}

EdgeId MeshTopology::sharedVertInOrg( FaceId l, FaceId r ) const
{
    assert( l && r && l != r );
    VertId vid[3];
    getLeftTriVerts( edgePerFace_[l], vid );
    for ( auto v : vid )
        for ( auto e : orgRing( *this, v ) )
            if ( left( e ) == r )
                return e;
    return {};
}

FaceId MeshTopology::sharedFace( EdgeId a, EdgeId b ) const
{
    assert( a && b );
    const auto al = left( a );
    const auto bl = left( b );
    if ( al && al == bl )
        return al;
    const auto ar = right( a );
    if ( ar && ar == bl )
        return ar;
    const auto br = right( b );
    if ( al && al == br )
        return al;
    if ( ar && ar == br )
        return ar;
    return {};
}

FaceId MeshTopology::lastValidFace() const
{
    if ( numValidFaces_ <= 0 )
        return {};
    for ( FaceId i{ (int)validFaces_.size() - 1 }; i.valid(); --i )
    {
        if ( validFaces_.test( i ) )
            return i;
    }
    assert( false );
    return {};
}

void MeshTopology::deleteFace( FaceId f )
{
    EdgeId e = edgeWithLeft( f );
    assert( e.valid() );
    if ( !e.valid() )
        return;

    // delete the face itself
    setLeft( e, FaceId{} );

    // delete not shared vertices and edges
    const int d = getLeftDegree( e );
    for ( int i = 0; i < d; ++i )
    {
        if ( !right( e ).valid() && prev( e ) == next( e ) )
        {
            // only two edges from e.origin, so this vertex does not belong to any other face
            setOrg( e, VertId{} );
        }
        EdgeId e1 = e;
        e = prev( e.sym() );
        if ( !left( e1.sym() ).valid() )
        {
            // no face to the right of e1, delete it
            splice( prev( e1 ), e1 );
            splice( prev( e1.sym() ), e1.sym() );
        }
    }
}

void MeshTopology::deleteFaces( const FaceBitSet& fs )
{
    MR_TIMER
    for ( auto f : fs )
        deleteFace( f );
}

void MeshTopology::translateNoFlip_( HalfEdgeRecord & r, const FaceMap & fmap, const VertMap & vmap, const WholeEdgeMap & emap ) const
{
    for ( auto n = r.next; ; n = next( n ) )
    {
        if ( (  r.next = mapEdge( emap, n ) ) )
            break;
    }

    for ( auto p = r.prev; ; p = prev( p ) )
    {
        if ( ( r.prev = mapEdge( emap, p ) ) )
            break;
    }

    if ( r.org.valid() )
        r.org = vmap[r.org];

    if ( r.left.valid() )
        r.left = fmap[r.left];
}

void MeshTopology::translate_( HalfEdgeRecord & r, HalfEdgeRecord & rsym,
    const FaceMap & fmap, const VertMap & vmap, const WholeEdgeMap & emap, bool flipOrientation ) const
{
    translateNoFlip_( r, fmap, vmap, emap );
    translateNoFlip_( rsym, fmap, vmap, emap );

    if ( flipOrientation )
    {
        std::swap( r.prev, r.next );
        std::swap( rsym.prev, rsym.next );
        std::swap( r.left, rsym.left );
    }
}

void MeshTopology::translateNoFlip_( HalfEdgeRecord & r, const FaceHashMap & fmap, const VertHashMap & vmap, const WholeEdgeHashMap & emap ) const
{
    for ( auto n = r.next; ; n = next( n ) )
    {
        if ( (  r.next = mapEdge( emap, n ) ) )
            break;
    }

    for ( auto p = r.prev; ; p = prev( p ) )
    {
        if ( ( r.prev = mapEdge( emap, p ) ) )
            break;
    }

    if ( r.org.valid() )
        r.org = getAt( vmap, r.org );

    if ( r.left.valid() )
        r.left = getAt( fmap, r.left );
}

void MeshTopology::translate_( HalfEdgeRecord & r, HalfEdgeRecord & rsym,
    const FaceHashMap & fmap, const VertHashMap & vmap, const WholeEdgeHashMap & emap, bool flipOrientation ) const
{
    translateNoFlip_( r, fmap, vmap, emap );
    translateNoFlip_( rsym, fmap, vmap, emap );

    if ( flipOrientation )
    {
        std::swap( r.prev, r.next );
        std::swap( rsym.prev, rsym.next );
        std::swap( r.left, rsym.left );
    }
}

void MeshTopology::flipEdge( EdgeId e )
{
    assert( isLeftTri( e ) );
    assert( isLeftTri( e.sym() ) );

    FaceId l = left( e );
    FaceId r = right( e );
    setLeft_( e, FaceId{} );
    setLeft_( e.sym(), FaceId{} );

    EdgeId a = next( e.sym() ).sym();
    EdgeId b = next( e ).sym();
    splice( prev( e ), e );
    splice( prev( e.sym() ), e.sym() );
    splice( a, e );
    splice( b, e.sym() );

    setLeft_( e, l );
    setLeft_( e.sym(), r );

    if ( l.valid() )
        edgePerFace_[l] = e;
    if ( r.valid() )
        edgePerFace_[r] = e.sym();
}

EdgeId MeshTopology::splitEdge( EdgeId e, FaceBitSet * region )
{
    FaceId l = left( e );
    if ( l.valid() )
    {
        assert( isLeftTri( e ) );
        setLeft_( e, FaceId{} );
    }
    FaceId r = right( e );
    if ( r.valid() )
    {
        assert( isLeftTri( e.sym() ) );
        setLeft_( e.sym(), FaceId{} );
    }
    
    // disconnect edge e from its origin
    EdgeId ePrev = prev( e );
    VertId v0;
    if ( ePrev != e )
    {
        splice( ePrev, e );
    }
    else
    {
        v0 = org( e );
        setOrg( e, {} );
    }

    // e now becomes the second part of split edge, add first part to it
    EdgeId e0 = makeEdge();
    assert( !org(e) );
    splice( e, e0.sym() );
    if ( ePrev != e )
        splice( ePrev, e0 );
    else
        setOrg( e0, v0 );

    // subdivide left and right triangles
    EdgeId eSymPrev = prev( e.sym() );
    if ( l.valid() && e.sym() != eSymPrev )
    {
        EdgeId el = makeEdge();
        splice( e, el );
        splice( prev( eSymPrev.sym() ), el.sym() );
        auto newFace = addFaceId();
        setLeft( el, newFace );
        if ( region && region->test( l ) )
            region->autoResizeSet( newFace );
    }
    if ( r.valid() && ePrev != e )
    {
        EdgeId er = makeEdge();
        splice( e0.sym(), er );
        splice( prev( ePrev.sym() ), er.sym() );
        auto newFace = addFaceId();
        setLeft( er.sym(), newFace );
        if ( region && region->test( r ) )
            region->autoResizeSet( newFace );
    }

    setLeft_( e, l );
    setLeft_( e.sym(), r );

    if ( l.valid() )
        edgePerFace_[l] = e;
    if ( r.valid() )
        edgePerFace_[r] = e.sym();

    // allocate id from new vertex
    VertId newv = addVertId();
    setOrg( e, newv );
    return e0;
}

VertId MeshTopology::splitFace( FaceId f, FaceBitSet * region )
{
    assert( !region || region->test( f ) );

    EdgeId e[3];
    e[0] = edgeWithLeft( f );
    assert( isLeftTri( e[0] ) );
    e[1] = prev( e[0].sym() );
    e[2] = prev( e[1].sym() );

    setLeft_( e[0], FaceId{} );

    EdgeId n[3];
    for ( int i = 0; i < 3; ++i )
    {
        n[i] = makeEdge();
        splice( e[i], n[i] );
    }

    // connect new edges in new vertex
    splice( n[0].sym(), n[1].sym() );
    splice( n[1].sym(), n[2].sym() );
    VertId newv = addVertId();
    setOrg( n[0].sym(), newv );

    for ( int i = 0; i < 3; ++i )
    {
        assert( isLeftTri( e[i] ) );
    }

    setLeft_( e[0], f );
    const auto f1 = addFaceId();
    setLeft( e[1], f1 );
    const auto f2 = addFaceId();
    setLeft( e[2], f2 );

    if ( region )
    {
        region->autoResizeSet( f1 );
        region->autoResizeSet( f2 );
    }

    return newv;
}

void MeshTopology::flipOrientation()
{
    MR_TIMER

    for ( auto & e : edgePerFace_ )
    {
        if ( e.valid() )
            e = e.sym();
    }

    for ( EdgeId i{0}; i + 1 < edges_.size(); ++++i )
    {
        auto & r0 = edges_[i];
        std::swap( r0.next, r0.prev );

        auto & r1 = edges_[i + 1];
        std::swap( r1.next, r1.prev );

        std::swap( r0.left, r1.left );
    }
}

void MeshTopology::addPart( const MeshTopology & from,
    FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap, bool rearrangeTriangles )
{
    MR_TIMER

    // in all maps: from index -> to index
    WholeEdgeMap emap;
    emap.resize( from.undirectedEdgeSize() );
    EdgeId firstNewEdge = edges_.endId();
    for ( UndirectedEdgeId i{ 0 }; i < emap.size(); ++i )
    {
        if ( from.isLoneEdge( i ) )
            continue;
        emap[i] = edges_.endId();
        edges_.push_back( from.edges_[ EdgeId( i ) ] );
        edges_.push_back( from.edges_[ EdgeId( i ).sym() ] );
    }

    VertMap vmap;
    VertId lastFromValidVertId = from.lastValidVert();
    vmap.resize( lastFromValidVertId + 1 );
    for ( VertId i{ 0 }; i <= lastFromValidVertId; ++i )
    {
        auto efrom = from.edgePerVertex_[i];
        if ( !efrom.valid() )
            continue;
        auto nv = addVertId();
        vmap[i] = nv;
        edgePerVertex_[nv] = mapEdge( emap, efrom );
        validVerts_.set( nv );
        ++numValidVerts_;
    }

    FaceMap fmap;
    FaceId lastFromValidFaceId = from.lastValidFace();
    fmap.resize( lastFromValidFaceId + 1 );
    FaceId firstNewFace( (int)edgePerFace_.size() );

    if ( rearrangeTriangles )
    {
        FaceMap invMap;
        invMap.reserve( from.numValidFaces_ );
        for ( auto i : from.validFaces_ )
            invMap.push_back( i );

        // returns true if a-face has smaller vertex ids than b-face
        auto isFromFaceLess = [&]( FaceId af, FaceId bf )
        {
            auto a = from.edgeWithLeft( af );
            auto b = from.edgeWithLeft( bf );
            for ( int i = 0; i < 3; ++i ) // triangular faces are most interesting
            {
                auto av = from.org( a );
                auto bv = from.org( b );
                if ( av != bv )
                    return av < bv;
                a = from.next( a.sym() );
                b = from.next( b.sym() );
            }
            return false;
        };

        std::sort( begin( invMap ), end( invMap ), isFromFaceLess );
        for ( auto i : invMap )
            fmap[i] = addFaceId();
    }
    else
    {
        for ( auto i : from.validFaces_ )
            fmap[i] = addFaceId();
    }

    for ( FaceId i{ 0 }; i <= lastFromValidFaceId; ++i )
    {
        auto efrom = from.edgePerFace_[i];
        if ( !efrom.valid() )
            continue;
        auto nf = fmap[i];
        edgePerFace_[nf] = mapEdge( emap, efrom );
    }
    validFaces_.set( firstNewFace, from.numValidFaces_, true );
    numValidFaces_ += from.numValidFaces_;

    // translate edge records
    tbb::parallel_for( tbb::blocked_range( firstNewEdge.undirected(), edges_.endId().undirected() ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            EdgeId e{ ue };
            from.translate_( edges_[e], edges_[e.sym()], fmap, vmap, emap, false );
        }
    } );

    if ( outFmap )
        *outFmap = std::move( fmap );
    if ( outVmap )
        *outVmap = std::move( vmap );
    if ( outEmap )
        *outEmap = std::move( emap );
}

bool MeshTopology::operator ==( const MeshTopology & b ) const
{
    MR_TIMER
    // make fast comparisons first
    if ( numValidVerts_ != b.numValidVerts_
        || numValidFaces_ != b.numValidFaces_
        || validVerts_ != b.validVerts_
        || validFaces_ != b.validFaces_ )
        return false;

    /* uncommenting this breaks MeshDiff unit test
    for ( auto v : validVerts_ )
        if ( edgePerVertex_[v] != b.edgePerVertex_[v] )
            return false;

    for ( auto f : validFaces_ )
        if ( edgePerFace_[f] != b.edgePerFace_[f] )
            return false;
    */

    return edges_ == b.edges_;
}

template<class V>
void resizeNoInit( V & v, size_t targetSize )
{
    // allocate enough memory
    v.reserve( targetSize );
    // resize without memory access
    while ( v.size() < targetSize )
        v.emplace_back( noInit );
    // in case initial size was larger
    v.resize( targetSize );
}

void MeshTopology::resizeBeforeParallelAdd( size_t edgeSize, size_t vertSize, size_t faceSize )
{
    MR_TIMER

    resizeNoInit( edges_, edgeSize );

    edgePerVertex_.resize( vertSize );
    validVerts_.resize( vertSize );

    edgePerFace_.resize( faceSize );
    validFaces_.resize( faceSize );
}

void MeshTopology::addPackedPart( const MeshTopology & from, EdgeId toEdgeId, const FaceMap & fmap, const VertMap & vmap )
{
    MR_TIMER

    assert( toEdgeId.valid() );
    assert( (int)toEdgeId + from.edges_.size() <= edges_.size() );
    // in all maps: from index -> to index
    auto emap = [toEdgeId]( EdgeId e ) { return toEdgeId + (int)e; };

    VertId lastFromValidVertId = from.lastValidVert();
    assert( (int)vmap.size() > lastFromValidVertId );
    for ( VertId i{ 0 }; i <= lastFromValidVertId; ++i )
    {
        auto efrom = from.edgePerVertex_[i];
        if ( !efrom.valid() )
            continue;
        auto & ev = edgePerVertex_[vmap[i]];
        assert( !ev.valid() );
        ev = emap( efrom );
    }

    FaceId lastFromValidFaceId = from.lastValidFace();
    assert( (int)fmap.size() > lastFromValidFaceId );
    for ( FaceId i{ 0 }; i <= lastFromValidFaceId; ++i )
    {
        auto efrom = from.edgePerFace_[i];
        if ( !efrom.valid() )
            continue;
        auto & ev = edgePerFace_[fmap[i]];
        assert( !ev.valid() );
        ev = emap( efrom );
    }

    // translate edge records
    for ( EdgeId i{ 0 }; i < from.edgeSize(); ++i )
    {
        assert ( !from.isLoneEdge( i ) );

        const HalfEdgeRecord & fromEdge = from.edges_[i];
        HalfEdgeRecord & to = edges_[emap( i )];
        to.next = emap( fromEdge.next );
        to.prev = emap( fromEdge.prev );
        to.org = fromEdge.org.valid() ? vmap[fromEdge.org] : VertId{};
        to.left = fromEdge.left.valid() ? fmap[fromEdge.left] : FaceId{};
    }
}

void MeshTopology::computeValidsFromEdges()
{
    MR_TIMER

    BitSetParallelForAll( validVerts_, [&]( VertId v )
    {
        if ( edgePerVertex_[v].valid() )
            validVerts_.set( v );
    } );

    numValidVerts_ = parallel_reduce( tbb::blocked_range( 0_v, VertId{ vertSize() } ), 0,
    [&] ( const auto & range, int curr )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
            if ( validVerts_.test( v ) )
                ++curr;
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );

    BitSetParallelForAll( validFaces_, [&]( FaceId f )
    {
        if ( edgePerFace_[f].valid() )
            validFaces_.set( f );
    } );

    numValidFaces_ = parallel_reduce( tbb::blocked_range( 0_f, FaceId{ faceSize() } ), 0,
    [&] ( const auto & range, int curr )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
            if ( validFaces_.test( f ) )
                ++curr;
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
}

void MeshTopology::computeAllFromEdges_()
{
    MR_TIMER

    VertId maxValidVert;
    FaceId maxValidFace;
    for( const auto & he : edges_ )
    {
        maxValidVert = std::max( maxValidVert, he.org );
        maxValidFace = std::max( maxValidFace, he.left );
    }

    edgePerVertex_.clear();
    edgePerVertex_.resize( maxValidVert + 1 );
    validVerts_.clear();
    validVerts_.resize( maxValidVert + 1 );
    numValidVerts_ = 0;

    edgePerFace_.clear();
    edgePerFace_.resize( maxValidFace + 1 );
    validFaces_.clear();
    validFaces_.resize( maxValidFace + 1 );
    numValidFaces_ = 0;

    for ( EdgeId e{0}; e < edges_.size(); ++e )
    {
        const auto & he = edges_[e];
        if ( he.org.valid() )
        {
            if ( !validVerts_.test_set( he.org ) )
            {
                edgePerVertex_[he.org] = e;
                ++numValidVerts_;
            }
        }
        if ( he.left.valid() )
        {
            if ( !validFaces_.test_set( he.left ) )
            {
                edgePerFace_[he.left] = e;
                ++numValidFaces_;
            }
        }
    }
}

void MeshTopology::addPartByMask( const MeshTopology & from, const FaceBitSet & fromFaces, const PartMapping & map )
{
    addPartByMask( from, fromFaces, false, {}, {}, map );
}

void MeshTopology::addPartByMask( const MeshTopology & from, const FaceBitSet & fromFaces, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    const PartMapping & map )
{
    MR_TIMER
    addPartBy( from, begin( fromFaces ), end( fromFaces ), fromFaces.count(), flipOrientation, thisContours, fromContours, map );
}

void MeshTopology::addPartByFaceMap( const MeshTopology & from, const FaceMap & fromFaces, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours, const std::vector<std::vector<EdgeId>> & fromContours,
    const PartMapping & map )
{
    MR_TIMER
    addPartBy( from, begin( fromFaces ), end( fromFaces ), fromFaces.size(), flipOrientation, thisContours, fromContours, map );
}

template<typename I>
void MeshTopology::addPartBy( const MeshTopology & from, I fbegin, I fend, size_t fcount, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    const PartMapping & map )
{
    MR_TIMER
    const auto szContours = thisContours.size();
    assert( szContours == fromContours.size() );
    
    auto set = []( auto & map, auto key, auto val )
    {
        auto [it, inserted] = map.insert( std::make_pair( key, val ) );
        if ( !inserted )
            assert( it->second == val );
    };

    // in all maps: from index -> to index
    FaceHashMap fmap;
    fmap.reserve( fcount );
    WholeEdgeHashMap emap;
    emap.reserve( std::min( 2 * fcount, from.undirectedEdgeSize() ) ); // if whole connected component is copied then ecount=3/2*fcount; if unconnected triangles are copied then ecount=3*fcount
    VertHashMap vmap;
    vmap.reserve( std::min( fcount, from.vertSize() ) ); // if whole connected component is copied then vcount=1/2*fcount; if unconnected triangles are copied then vcount=3*fcount
    if ( map.tgt2srcEdges )
        map.tgt2srcEdges->resize( undirectedEdgeSize() );
    if ( map.tgt2srcVerts )
        map.tgt2srcVerts->resize( vertSize() );
    if ( map.tgt2srcFaces )
        map.tgt2srcFaces->resize( faceSize() );

    UndirectedEdgeBitSet existingEdges; //one of fromContours' edge
    for ( int i = 0; i < szContours; ++i )
    {
        const auto & thisContour = thisContours[i];
        const auto & fromContour = fromContours[i];
        const auto sz = thisContour.size();
        assert( sz == fromContour.size() );
        // either both contours are closed or both are open
        [[maybe_unused]] auto s0 = from.org( fromContour.front() );
        [[maybe_unused]] auto t0 = from.dest( fromContour.back() );
        [[maybe_unused]] auto s1 = org( thisContour.front() );
        [[maybe_unused]] auto t1 = dest( thisContour.back() );
        assert( ( s0 == t0 && s1 == t1 ) || ( s0 != t0 && s1 != t1 ) );
        for ( int j = 0; j < sz; ++j )
        {
            auto e = fromContour[j];
            auto e1 = thisContour[j];
            assert( !left( e1 ) );
            assert( ( flipOrientation && !from.left( e ) ) || ( !flipOrientation && !from.right( e ) ) );
            set( vmap, from.org( e ), org( e1 ) );
            set( vmap, from.dest( e ), dest( e1 ) );
            set( emap, e.undirected(), e.even() ? e1 : e1.sym() );
            existingEdges.autoResizeSet( e.undirected() );
        }
    }

    // first pass: fill maps
    EdgeId firstNewEdge = edges_.endId();
    for ( ; fbegin != fend; ++fbegin )
    {
        auto f = *fbegin;
        auto efrom = from.edgePerFace_[f];
        for ( auto e : leftRing( from, efrom ) )
        {
            const UndirectedEdgeId ue = e.undirected();
            if ( auto [it, inserted] = emap.insert( { ue, {} } ); inserted )
            {
                it->second = edges_.endId();
                edges_.push_back( from.edges_[EdgeId{ue}] );
                edges_.push_back( from.edges_[EdgeId{ue}.sym()] );
                if ( map.tgt2srcEdges )
                {
                    map.tgt2srcEdges ->push_back( EdgeId{ue} );
                }
            }
            if ( auto v = from.org( e ); v.valid() )
            {
                if ( auto [it, inserted] = vmap.insert( { v, {} } ); inserted )
                {
                    auto nv = addVertId();
                    if ( map.tgt2srcVerts )
                        map.tgt2srcVerts ->push_back( v );
                    it->second = nv;
                    edgePerVertex_[nv] = mapEdge( emap, e );
                    validVerts_.set( nv );
                    ++numValidVerts_;
                }
            }
        }
        auto nf = addFaceId();
        if ( map.tgt2srcFaces )
            map.tgt2srcFaces ->push_back( f );
        fmap[f] = nf;
        edgePerFace_[nf] = mapEdge( emap, flipOrientation ? efrom.sym() : efrom );
        validFaces_.set( nf );
        ++numValidFaces_;
    }

    // in case of open contours, some nearby edges have to be updated
    std::vector<std::pair<EdgeId, EdgeId>> prevNextEdges;
    for ( int i = 0; i < szContours; ++i )
    {
        const auto & thisContour = thisContours[i];
        const auto & fromContour = fromContours[i];
        const auto sz = thisContour.size();
        for ( int j = 0; j < sz; ++j )
        {
            auto e = fromContour[j];
            auto e1 = thisContour[j];

            auto eNx = flipOrientation ? from.prev( e.sym() ) : from.next( e.sym() );
            if ( !existingEdges.test( eNx.undirected() ) )
            {
                auto e1Nx = prev( e1.sym() );
                prevNextEdges.emplace_back( e1Nx, mapEdge( emap, eNx ) );
            }

            auto ePr = flipOrientation ? from.next( e ) : from.prev( e );
            if ( !existingEdges.test( ePr.undirected() ) )
            {
                auto e1Pr = next( e1 );
                prevNextEdges.emplace_back( mapEdge( emap, ePr ), e1Pr );
            }
        }
    }

    // update records of stitched edges
    for ( int i = 0; i < szContours; ++i )
    {
        const auto & thisContour = thisContours[i];
        const auto & fromContour = fromContours[i];
        const auto sz = thisContour.size();
        assert( sz == fromContour.size() );
        for ( int j = 0; j < sz; ++j )
        {
            auto e = fromContour[j];
            auto e1 = thisContour[j];

            {
                assert ( existingEdges.test( e.undirected() ) );
                assert( !left( e1 ) );
                HalfEdgeRecord & toHe = edges_[e1];
                const HalfEdgeRecord & fromHe = from.edges_[e];
                toHe.next = mapEdge( emap, flipOrientation ? fromHe.prev : fromHe.next );
                assert( toHe.next );
                if ( auto left = flipOrientation ? from.edges_[e.sym()].left : fromHe.left )
                    toHe.left = getAt( fmap, left );
            }
            {
                HalfEdgeRecord & toHe = edges_[e1.sym()];
                const HalfEdgeRecord & fromHe = from.edges_[e.sym()];
                toHe.prev = mapEdge( emap, flipOrientation ? fromHe.next : fromHe.prev );
                assert( toHe.prev );
            }
        }
    }

    // second pass: translate edge records
    tbb::parallel_for( tbb::blocked_range( firstNewEdge.undirected(), edges_.endId().undirected() ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            EdgeId e{ ue };
            from.translate_( edges_[e], edges_[e.sym()], fmap, vmap, emap, flipOrientation );
        }
    } );

    // update near stitch edges
    for ( const auto& [ePr, eNx] : prevNextEdges )
    {
        assert( org( ePr ) == org( eNx ) );
        assert( !left( ePr ) );
        assert( !right( eNx ) );

        edges_[ePr].next = eNx;
        edges_[eNx].prev = ePr;
    }

    if ( map.tgt2srcEdges )
        assert( map.tgt2srcEdges->size() == undirectedEdgeSize() );
    if ( map.tgt2srcVerts )
        assert( map.tgt2srcVerts->size() == vertSize() );
    if ( map.tgt2srcFaces )
        assert( map.tgt2srcFaces->size() == faceSize() );

    if ( map.src2tgtFaces )
        *map.src2tgtFaces = std::move( fmap );
    if ( map.src2tgtVerts )
        *map.src2tgtVerts = std::move( vmap );
    if ( map.src2tgtEdges )
        *map.src2tgtEdges = std::move( emap );
}

template MRMESH_API void MeshTopology::addPartBy( const MeshTopology & from,
    SetBitIteratorT<FaceBitSet> fbegin, SetBitIteratorT<FaceBitSet> fend, size_t fcount, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    const PartMapping & map );
template MRMESH_API void MeshTopology::addPartBy( const MeshTopology & from,
    FaceMap::iterator fbegin, FaceMap::iterator fend, size_t fcount, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    const PartMapping & map );

void MeshTopology::rotateTriangles()
{
    MR_TIMER

    tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId{0}, FaceId{edgePerFace_.size()} ), [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
        {
            EdgeId emin = edgePerFace_[f];
            if ( !emin.valid() )
                continue;
            VertId vmin = org( emin );
            for ( EdgeId e : leftRing0( *this, emin ) )
            {
                VertId v = org( e );
                if ( v < vmin )
                {
                    vmin = v;
                    emin = e;
                }
            }
            edgePerFace_[f] = emin;
        }
    } );
}

void MeshTopology::pack( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap, bool rearrangeTriangles )
{
    MR_TIMER

    if ( rearrangeTriangles )
        rotateTriangles();
    MeshTopology packed;
    packed.addPart( *this, outFmap, outVmap, outEmap, rearrangeTriangles );
    *this = std::move( packed );
}

inline EdgeId getAt( const Buffer<UndirectedEdgeId, UndirectedEdgeId> & bmap, EdgeId key )
{
    EdgeId res;
    if ( key )
    {
        res = bmap[ key.undirected() ];
        if ( key.odd() )
            res = res.sym();
    }
    return res;
}

void MeshTopology::pack( const PackMapping & map )
{
    MR_TIMER

    Vector<HalfEdgeRecord, EdgeId> newEdges;
    resizeNoInit( newEdges, 2 * map.e.tsize );
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( undirectedEdgeSize() ) ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( auto oldUe = range.begin(); oldUe < range.end(); ++oldUe )
        {
            auto newUe = map.e.b[oldUe];
            if ( !newUe )
                continue;
            auto translateHalfEdge = [&]( const HalfEdgeRecord & he )
            {
                HalfEdgeRecord res;
                res.next = getAt( map.e.b, he.next );
                res.prev = getAt( map.e.b, he.prev );
                res.org = getAt( map.v.b, he.org );
                res.left = getAt( map.f.b, he.left );
                return res;
            };
            newEdges[ EdgeId{newUe} ] = translateHalfEdge( edges_[ EdgeId{oldUe} ] );
            newEdges[ EdgeId{newUe}.sym() ] = translateHalfEdge( edges_[ EdgeId{oldUe}.sym() ] );
        }
    } );
    edges_ = std::move( newEdges );

    Vector<EdgeId, FaceId> newEdgePerFace;
    resizeNoInit( newEdgePerFace, map.f.tsize );
    tbb::parallel_for( tbb::blocked_range( 0_f, FaceId( faceSize() ) ),
        [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( auto oldf = range.begin(); oldf < range.end(); ++oldf )
        {
            auto newf = map.f.b[oldf];
            if ( !newf )
                continue;
            newEdgePerFace[newf] = getAt( map.e.b, edgePerFace_[oldf] );
        }
    } );
    edgePerFace_ = std::move( newEdgePerFace );
    assert( edgePerFace_.size() == numValidFaces_ );
    validFaces_.resize( edgePerFace_.size() );
    validFaces_.set( 0_f, edgePerFace_.size(), true );

    Vector<EdgeId, VertId> newEdgePerVertex;
    resizeNoInit( newEdgePerVertex, map.v.tsize );
    tbb::parallel_for( tbb::blocked_range( 0_v, VertId( vertSize() ) ),
        [&]( const tbb::blocked_range<VertId> & range )
    {
        for ( auto oldv = range.begin(); oldv < range.end(); ++oldv )
        {
            auto newv = map.v.b[oldv];
            if ( !newv )
                continue;
            newEdgePerVertex[newv] = getAt( map.e.b, edgePerVertex_[oldv] );
        }
    } );
    edgePerVertex_ = std::move( newEdgePerVertex );
    assert( edgePerVertex_.size() == numValidVerts_ );
    validVerts_.resize( edgePerVertex_.size() );
    validVerts_.set( 0_v, edgePerVertex_.size(), true );
}

void MeshTopology::write( std::ostream & s ) const
{
    // write edges
    auto numEdges = (std::uint32_t)edges_.size();
    s.write( (const char*)&numEdges, 4 );
    s.write( (const char*)edges_.data(), edges_.size() * sizeof(HalfEdgeRecord) );

    // write verts
    auto numVerts = (std::uint32_t)edgePerVertex_.size();
    s.write( (const char*)&numVerts, 4 );
    s.write( (const char*)edgePerVertex_.data(), edgePerVertex_.size() * sizeof(EdgeId) );

    // write faces
    auto numFaces = (std::uint32_t)edgePerFace_.size();
    s.write( (const char*)&numFaces, 4 );
    s.write( (const char*)edgePerFace_.data(), edgePerFace_.size() * sizeof(EdgeId) );
}

tl::expected<void, std::string> MeshTopology::read( std::istream & s, ProgressCallback callback )
{
    // read edges
    std::uint32_t numEdges;
    s.read( (char*)&numEdges, 4 );
    if ( !s )
        return tl::make_unexpected( std::string( "Stream reading error" ) );

    auto posCur = s.tellg();
    s.seekg( 0, std::ios_base::end );
    const auto posEnd = s.tellg();
    s.seekg( posCur );
    if ( size_t( posEnd - posCur ) < numEdges * sizeof( HalfEdgeRecord ) )
        return tl::make_unexpected( std::string( "Stream reading error: stream is too short" ) ); // stream is too short

    edges_.resize( numEdges );
    if ( !readByBlocks( s, ( char* )edges_.data(), edges_.size() * sizeof( HalfEdgeRecord ),
        callback ? [callback] ( float v )
    {
        return callback( v / 3.f );
    } : callback ) )
        return tl::make_unexpected( std::string( "Loading canceled" ) );

    // read verts
    std::uint32_t numVerts;
    s.read( (char*)&numVerts, 4 );
    if ( !s )
        return tl::make_unexpected( std::string( "Stream reading error" ) );
    edgePerVertex_.resize( numVerts );
    validVerts_.resize( numVerts );
    if ( !readByBlocks( s, (char*)edgePerVertex_.data(), edgePerVertex_.size() * sizeof( EdgeId ),
        callback ? [callback] ( float v )
    {
        return callback( ( 1.f + v ) / 3.f );
    } : callback ) )
        return tl::make_unexpected( std::string( "Loading canceled" ) );

    // read faces
    std::uint32_t numFaces;
    s.read( (char*)&numFaces, 4 );
    if ( !s )
        return tl::make_unexpected( std::string( "Stream reading error" ) );
    edgePerFace_.resize( numFaces );
    validFaces_.resize( numFaces );
    if ( !readByBlocks( s, (char*)edgePerFace_.data(), edgePerFace_.size() * sizeof( EdgeId ),
        callback ? [callback] ( float v )
    {
        return callback( ( 2.f + v ) / 3.f );
    } : callback ) )
        return tl::make_unexpected( std::string( "Loading canceled" ) );

    computeValidsFromEdges();

    if ( !s.good() )
        return tl::make_unexpected( std::string( "Stream reading error" ) );
    if ( !checkValidity() )
        return tl::make_unexpected( std::string( "Data is invalid" ) );
    return {};
}

#define CHECK(x) { assert(x); if (!(x)) return false; }

bool MeshTopology::checkValidity() const
{
    MR_TIMER

    for ( EdgeId e{0}; e < edges_.size(); ++e )
    {
        CHECK( edges_[edges_[e].next].prev == e );
        CHECK( edges_[edges_[e].prev].next == e );
        if ( auto v = edges_[e].org )
            CHECK( validVerts_.test( v ) );
        if ( auto f = edges_[e].left )
            CHECK( validFaces_.test( f ) );
    }

    const auto vSize = edgePerVertex_.size();
    CHECK( vSize == validVerts_.size() )

    int realValidVerts = 0;
    for ( VertId v{0}; v < edgePerVertex_.size(); ++v )
    {
        if ( edgePerVertex_[v].valid() )
        {
            CHECK( validVerts_.test( v ) )
            CHECK( edgePerVertex_[v] < edges_.size() );
            CHECK( edges_[edgePerVertex_[v]].org == v );
            ++realValidVerts;
            for ( EdgeId e : orgRing( *this, v ) )
                CHECK( org(e) == v );
        }
        else
        {
            CHECK( !validVerts_.test( v ) )
        }
    }
    CHECK( numValidVerts_ == realValidVerts );

    const auto fSize = edgePerFace_.size();
    CHECK( fSize == validFaces_.size() )

    int realValidFaces = 0;
    for ( FaceId f{0}; f < edgePerFace_.size(); ++f )
    {
        if ( edgePerFace_[f].valid() )
        {
            CHECK( validFaces_.test( f ) )
            CHECK( edgePerFace_[f] < edges_.size() );
            CHECK( edges_[edgePerFace_[f]].left == f );
            ++realValidFaces;
            for ( EdgeId e : leftRing( *this, f ) )
                CHECK( left(e) == f );
        }
        else
        {
            CHECK( !validFaces_.test( f ) )
        }
    }
    CHECK( numValidFaces_ == realValidFaces );

    return true;
}

void loadMeshDll()
{
}

} //namespace MR
