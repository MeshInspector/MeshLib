#include "MRMeshTopology.h"
#include "MRRingIterator.h"
#include "MRRegionBoundary.h"
#include "MREdgeIterator.h"
#include "MREdgePaths.h"
#include "MRBuffer.h"
#include "MRMapEdge.h"
#include "MRNoDefInit.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRProgressReadWrite.h"
#include "MRGridSettings.h"
#include "MRIOParsing.h"
#include "MRPartMappingAdapters.h"
#include <atomic>
#include <initializer_list>

namespace MR
{

void MeshTopology::vertResize( size_t newSize )
{
    if ( edgePerVertex_.size() >= newSize )
        return;
    edgePerVertex_.resize( newSize );
    if ( updateValids_ )
        validVerts_.resize( newSize );
}

void MeshTopology::vertResizeWithReserve( size_t newSize )
{
    if ( edgePerVertex_.size() >= newSize )
        return;
    edgePerVertex_.resizeWithReserve( newSize );
    if ( updateValids_ )
        validVerts_.resizeWithReserve( newSize );
}

void MeshTopology::faceResize( size_t newSize )
{
    if ( edgePerFace_.size() >= newSize )
        return;
    edgePerFace_.resize( newSize );
    if ( updateValids_ )
        validFaces_.resize( newSize );
}

void MeshTopology::faceResizeWithReserve( size_t newSize )
{
    if ( edgePerFace_.size() >= newSize )
        return;
    edgePerFace_.resizeWithReserve( newSize );
    if ( updateValids_ )
        validFaces_.resizeWithReserve( newSize );
}

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

UndirectedEdgeId MeshTopology::lastNotLoneUndirectedEdge() const
{
    assert( edges_.size() % 2 == 0 );
    for ( UndirectedEdgeId i{ (int)undirectedEdgeSize() - 1 }; i.valid(); --i )
    {
        if ( !isLoneEdge( i ) )
            return i;
    }
    return {};
}

void MeshTopology::excludeLoneEdges( UndirectedEdgeBitSet & edges ) const
{
    MR_TIMER;
    for ( auto ue : edges )
        if ( isLoneEdge( ue ) )
            edges.reset( ue );
}

size_t MeshTopology::computeNotLoneUndirectedEdges() const
{
    MR_TIMER;

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

UndirectedEdgeBitSet MeshTopology::findNotLoneUndirectedEdges() const
{
    MR_TIMER;

    UndirectedEdgeBitSet res( undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        if ( !isLoneEdge( ue ) )
            res.set( ue );
    } );
    return res;
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

void MeshTopology::shrinkToFit()
{
    MR_TIMER;
    edges_.vec_.shrink_to_fit();
    edgePerVertex_.vec_.shrink_to_fit();
    validVerts_.shrink_to_fit();
    edgePerFace_.vec_.shrink_to_fit();
    validFaces_.shrink_to_fit();
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
        if ( !fromSameOriginRing( edgePerVertex_[aData.org], a ) )
            edgePerVertex_[aData.org] = a;
    }

    if ( wasSameLeftId && bData.left.valid() )
    {
        setLeft_( b, FaceId() );
        if ( !fromSameLeftRing( edgePerFace_[aData.left], a ) )
            edgePerFace_[aData.left] = a;
    }
}

EdgeId MeshTopology::collapseEdge( const EdgeId e, const std::function<void( EdgeId del, EdgeId rem )> & onEdgeDel )
{
    auto delEdge = [&]( EdgeId del )
    {
        assert( del );
        if ( onEdgeDel )
            onEdgeDel( del, {} );
    };
    auto replaceEdge = [&]( EdgeId del, EdgeId rem )
    {
        assert( del && rem );
        if ( onEdgeDel )
            onEdgeDel( del, rem );
    };

    setLeft( e, FaceId() );
    setLeft( e.sym(), FaceId() );

    delEdge( e );

    if ( next( e ) == e )
    {
        setOrg( e, VertId() );
        const EdgeId b = prev( e.sym() );
        if ( b == e.sym() )
            setOrg( e.sym(), VertId() );
        else
            splice( b, e.sym() );

        assert( isLoneEdge( e ) );
        return EdgeId();
    }

    setOrg( e.sym(), VertId() );

    const EdgeId ePrev = prev( e );
    const EdgeId eNext = next( e );
    if ( ePrev != e )
        splice( ePrev, e );

    const EdgeId a = next( e.sym() );
    if ( a == e.sym() )
    {
        assert( isLoneEdge( e ) );
        return ePrev != e ? ePrev : EdgeId();
    }
    const EdgeId b = prev( e.sym() );

    splice( b, e.sym() );
    assert( isLoneEdge( e ) );

    assert( next( b ) == a );
    assert( next( ePrev ) == eNext );
    splice( b, ePrev );
    assert( next( b ) == eNext );
    assert( next( ePrev ) == a );

    if ( next( a.sym() ) == ePrev.sym() )
    {
        splice( ePrev, a );
        splice( prev( a.sym() ), a.sym() );
        assert( isLoneEdge( a ) );
        if ( !left( ePrev ) && !right( ePrev ) )
        {
            splice( prev( ePrev ), ePrev );
            splice( prev( ePrev.sym() ), ePrev.sym() );
            setOrg( ePrev, {} );
            setOrg( ePrev.sym(), {} );
            assert( isLoneEdge( ePrev ) );
            delEdge( a );
            delEdge( ePrev );
        }
        else
            replaceEdge( a, ePrev );
    }

    if ( next( eNext.sym() ) == b.sym() )
    {
        splice( eNext.sym(), b.sym() );
        splice( prev( b ), b );
        assert( isLoneEdge( b ) );
        if ( !left( eNext ) && !right( eNext ) )
        {
            splice( prev( eNext ), eNext );
            splice( prev( eNext.sym() ), eNext.sym() );
            setOrg( eNext, {} );
            setOrg( eNext.sym(), {} );
            assert( isLoneEdge( eNext ) );
            delEdge( b );
            delEdge( eNext );
        }
        else
            replaceEdge( b, eNext );
    }

    return ePrev != e ? ePrev : EdgeId();
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

int MeshTopology::getOrgDegree( EdgeId a ) const
{
    assert( a.valid() );
    int degree = 0;
    for( [[maybe_unused]] auto e : orgRing( *this, a ) )
        ++degree;
    return degree;
}

int MeshTopology::getLeftDegree( EdgeId a ) const
{
    assert( a.valid() );
    int degree = 0;
    for( [[maybe_unused]] auto e : leftRing( *this, a ) )
        ++degree;
    return degree;
}

bool MeshTopology::isLeftTri( EdgeId a ) const
{
    assert( a.valid() );
    EdgeId b = prev( a.sym() );
    // org(b) == dest(a)
    if ( a.sym() == b )
        return false;
    EdgeId c = prev( b.sym() );
    if ( a == c || b.sym() == c )
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

void MeshTopology::getLeftTriEdges( EdgeId e0, EdgeId & e1, EdgeId & e2 ) const
{
    e1 = prev( e0.sym() );
    e2 = prev( e1.sym() );
    assert( e0 == prev( e2.sym() ) );
}

std::vector<ThreeVertIds> MeshTopology::getAllTriVerts() const
{
    MR_TIMER;
    std::vector<ThreeVertIds> res;
    assert( updateValids_ );
    res.reserve( numValidFaces_ );
    for ( auto f : validFaces_ )
    {
        VertId vs[3];
        getTriVerts( f, vs );
        res.push_back( { vs[0], vs[1], vs[2] } );
    }

    return res;
}

Triangulation MeshTopology::getTriangulation() const
{
    MR_TIMER;
    Triangulation res;
    res.resize( faceSize() ); //TODO: resizeNoInit
    assert( updateValids_ );
    BitSetParallelFor( validFaces_, [&]( FaceId f )
    {
        getTriVerts( f, res[f] );
    } );

    return res;
}

bool MeshTopology::isLeftQuad( EdgeId a ) const
{
    assert( a.valid() );
    EdgeId b = prev( a.sym() );
    // org(b) == dest(a)
    if ( a.sym() == b )
        return false;
    EdgeId c = prev( b.sym() );
    if ( a == c || b.sym() == c )
        return false;
    EdgeId d = prev( c.sym() );
    if ( a == d || c.sym() == d )
        return false;
    EdgeId e = prev( d.sym() );
    return a == e;
}

EdgeId MeshTopology::bdEdgeSameLeft( EdgeId e, const FaceBitSet * region ) const
{
    for ( auto ei : leftRing( *this, e ) )
        if ( isBdEdge( ei, region ) )
            return ei;
    return {};
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
        if ( updateValids_ )
        {
            validVerts_.reset( oldV );
            --numValidVerts_;
        }
    }
    if ( v.valid() )
    {
        assert( !edgePerVertex_[v].valid() );
        edgePerVertex_[v] = a;
        if ( updateValids_ )
        {
            validVerts_.set( v );
            ++numValidVerts_;
        }
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
        if ( updateValids_ )
        {
            validFaces_.reset( oldF );
            --numValidFaces_;
        }
    }
    if ( f.valid() )
    {
        assert( !edgePerFace_[f].valid() );
        edgePerFace_[f] = a;
        if ( updateValids_ )
        {
            validFaces_.set( f );
            ++numValidFaces_;
        }
    }
}

bool MeshTopology::isInnerOrBdVertex( VertId v, const FaceBitSet * region ) const
{
    for ( auto e : orgRing( *this, v ) )
        if ( contains( region, left( e ) ) )
            return true;
    return false;
}

EdgeId MeshTopology::nextLeftBd( EdgeId e, const FaceBitSet * region ) const
{
    assert( isLeftBdEdge( e, region ) );

    for ( e = next( e.sym() ); !isLeftBdEdge( e, region ); e = next( e ) )
    {
        assert( !isLeftBdEdge( e.sym(), region ) );
    }
    return e;
}

EdgeId MeshTopology::prevLeftBd( EdgeId e, const FaceBitSet * region ) const
{
    assert( isLeftBdEdge( e, region ) );

    for ( e = prev( e ); !isLeftBdEdge( e.sym(), region ); e = prev( e ) )
    {
        assert( !isLeftBdEdge( e, region ) );
    }
    return e.sym();
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

bool MeshTopology::isClosed( const FaceBitSet * region ) const
{
    MR_TIMER;
    std::atomic_bool res{ true };
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( undirectedEdgeSize() ) ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            if ( !res.load( std::memory_order_relaxed ) )
                return;
            EdgeId e( ue );
            if ( isLoneEdge( e ) )
                continue;
            const auto l = left( e );
            const auto r = right( e );
            if ( l && r )
                continue; // no neighboring holes
            if ( region && !contains( *region, l ) && !contains( *region, r ) )
                continue; // both left and right are not in the region
            res.store( false, std::memory_order_relaxed );
            return;
        }
    } );
    return res.load( std::memory_order_relaxed );
}

std::vector<EdgeId> MeshTopology::findHoleRepresentiveEdges( const FaceBitSet * region ) const
{
    MR_TIMER;

    EdgeBitSet representativeEdges;
    const auto num = findNumHoles( &representativeEdges );

    std::vector<EdgeId> res;
    if ( num <= 0 )
        return res;

    res.reserve( num );
    for ( EdgeId e : representativeEdges )
        if ( !region || contains( *region, right( e ) ) )
            res.push_back( e );
    assert( region || res.size() == num );
    return res;
}

int MeshTopology::findNumHoles( EdgeBitSet * holeRepresentativeEdges ) const
{
    MR_TIMER;

    if ( holeRepresentativeEdges )
    {
        holeRepresentativeEdges->clear();
        holeRepresentativeEdges->resize( edgeSize(), false );
    }

    auto bdEdges = findLeftBdEdges();
    std::atomic<int> res;

    const int endBlock = int( bdEdges.size() + bdEdges.bits_per_block - 1 ) / bdEdges.bits_per_block;
    tbb::parallel_for( tbb::blocked_range<int>( 0, endBlock ),
        [&]( const tbb::blocked_range<int> & range )
        {
            int myHoles = 0; // with smallest edge in my range
            const EdgeId eBeg{ range.begin() * BitSet::bits_per_block };
            const EdgeId eEnd{ range.end() < endBlock ? range.end() * bdEdges.bits_per_block : bdEdges.size() };
            for ( auto e = eBeg; e < eEnd; ++e )
            {
                if ( !bdEdges.test( e ) )
                    continue;
                assert( !left( e ) );
                EdgeId smallestHoleEdge = e;
                for ( EdgeId ei : leftRing0( *this, e ) )
                {
                    if ( ei > e )
                    {
                        if ( ei < eEnd )
                        {
                            // skip this hole when its edge is encountered again,
                            // we can safely change only bits of our part
                            assert( bdEdges.test( ei ) );
                            bdEdges.reset( ei );
                            assert( !bdEdges.test( ei ) );
                        }
                    }
                    else if ( ei < smallestHoleEdge )
                        smallestHoleEdge = ei;
                }
                assert( smallestHoleEdge < eEnd );
                if ( smallestHoleEdge >= eBeg )
                {
                    ++myHoles;
                    if ( holeRepresentativeEdges )
                        holeRepresentativeEdges->set( smallestHoleEdge );
                }
            }
            res.fetch_add( myHoles, std::memory_order_relaxed );
        } );

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
    MR_TIMER;
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
    MR_TIMER;
    EdgeBitSet res( edges_.size() );
    BitSetParallelForAll( res, [&]( EdgeId e )
    {
        if ( !left( e ) && !isLoneEdge( e ) )
            res.set( e );
    } );
    return res;
}

bool MeshTopology::isBdEdge( EdgeId e, const FaceBitSet * region ) const
{
    if ( !region )
    {
        assert( !isLoneEdge( e ) );
        return !left( e ) || !right( e );
    }
    return isLeftInRegion( e, region ) != isLeftInRegion( e.sym(), region );
}

EdgeBitSet MeshTopology::findLeftBdEdges( const FaceBitSet * region, const EdgeBitSet * test ) const
{
    MR_TIMER;
    EdgeBitSet res( edges_.size() );
    BitSetParallelForAll( res, [&]( EdgeId e )
    {
        if ( test && !test->test( e ) )
            return;
        if ( !region && !left( e ) && !isLoneEdge( e ) )
            res.set( e );
        if ( isLeftInRegion( e.sym(), region ) && !isLeftInRegion( e, region ) ) // shall skip lone edges
            res.set( e );
    } );
    return res;
}

FaceBitSet MeshTopology::findBoundaryFaces( const FaceBitSet * region ) const
{
    MR_TIMER;
    const auto & fs = getFaceIds( region );
    FaceBitSet res( fs.size() );
    BitSetParallelFor( fs, [&]( FaceId f )
    {
        for ( EdgeId e : leftRing( *this, f ) )
        {
            if ( !right( e ) )
            {
                res.set( f );
                break;
            }
        }
    } );
    return res;
}

FaceBitSet MeshTopology::findBdFaces( const FaceBitSet * region ) const
{
    MR_TIMER;
    const auto & fs = getFaceIds( region );
    FaceBitSet res( fs.size() );
    BitSetParallelFor( fs, [&]( FaceId f )
    {
        if ( isBdFace( f, region ) )
            res.set( f );
    } );
    return res;
}

VertBitSet MeshTopology::findBoundaryVerts( const VertBitSet * region ) const
{
    MR_TIMER;
    const auto & vs = getVertIds( region );
    VertBitSet res( vs.size() );
    BitSetParallelFor( vs, [&]( VertId v )
    {
        for ( EdgeId e : orgRing( *this, v ) )
        {
            if ( !left( e ) )
            {
                res.set( v );
                break;
            }
        }
    } );
    return res;
}

VertBitSet MeshTopology::findBdVerts( const FaceBitSet * region, const VertBitSet * test ) const
{
    MR_TIMER;
    VertBitSet res( vertSize() );
    BitSetParallelFor( getVertIds( test ), [&]( VertId v )
    {
        if ( isBdVertex( v, region ) )
            res.set( v );
    } );
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
    assert( updateValids_ );
    if ( numValidVerts_ <= 0 )
        return {};
    return validVerts_.find_last();
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
    assert( updateValids_ );
    if ( numValidFaces_ <= 0 )
        return {};
    return validFaces_.find_last();
}

void MeshTopology::deleteFace( FaceId f, const UndirectedEdgeBitSet * keepEdges )
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
        EdgeId e1 = e;
        e = prev( e.sym() );
        if ( !right( e1 ) && !( keepEdges && keepEdges->test( e1 ) ) )
        {
            // delete e1 since it has no right face as well (and not in keepEdges);
            // also delete any end-vertex of e1 if e1 was the last edge from it
            if ( prev( e1 ) == e1 )
                setOrg( e1, VertId{} );
            else
                splice( prev( e1 ), e1 );

            if ( prev( e1.sym() ) == e1.sym() )
                setOrg( e1.sym(), VertId{} );
            else
                splice( prev( e1.sym() ), e1.sym() );

            assert( isLoneEdge( e1 ) );
        }
    }
}

void MeshTopology::deleteFaces( const FaceBitSet & fs, const UndirectedEdgeBitSet * keepEdges )
{
    MR_TIMER;
    for ( auto f : fs )
        deleteFace( f, keepEdges );
}

template<typename FM, typename VM, typename WEM>
void MeshTopology::translateNoFlip_( HalfEdgeRecord & r, const FM & fmap, const VM & vmap, const WEM & emap ) const
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

template<typename FM, typename VM, typename WEM>
void MeshTopology::translate_( HalfEdgeRecord & r, HalfEdgeRecord & rsym,
    const FM & fmap, const VM & vmap, const WEM & emap, bool flipOrientation ) const
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
    assert( !fromSameOriginRing( a, b ) ); //otherwise loop edge will appear
    splice( prev( e ), e );
    splice( prev( e.sym() ), e.sym() );
    splice( a, e );
    splice( b, e.sym() );

    assert( isLeftTri( e ) );
    assert( isLeftTri( e.sym() ) );

    setLeft_( e, l );
    setLeft_( e.sym(), r );

    if ( l.valid() )
        edgePerFace_[l] = e;
    if ( r.valid() )
        edgePerFace_[r] = e.sym();
}

static inline void setNewToOld( FaceHashMap * new2Old, std::initializer_list<FaceId> newFaces, FaceId fromFace )
{
    if ( !new2Old )
        return;
    if ( auto it = new2Old->find( fromFace ); it != new2Old->end() )
    {
        // fromFace is already new, find its origin
        fromFace = it->second;
        // now fromFace is original face that must not be found among new ones (not a key in new2Old)
        assert( new2Old->find( fromFace ) == new2Old->end() );
    }
    for ( auto newFace: newFaces )
        (*new2Old)[newFace] = fromFace;
}

EdgeId MeshTopology::splitEdge( EdgeId e, FaceBitSet * region, FaceHashMap * new2Old )
{
    FaceId l = left( e );
    if ( l.valid() )
        setLeft_( e, FaceId{} );
    FaceId r = right( e );
    if ( r.valid() )
        setLeft_( e.sym(), FaceId{} );

    // disconnect edge e from its origin
    const EdgeId ePrev = prev( e );
    const EdgeId eNext = next( e );
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

    // subdivide left and right faces
    if ( l.valid() && eNext != e )
    {
        EdgeId el = makeEdge();
        splice( e, el );
        splice( eNext.sym(), el.sym() );
        auto newFace = addFaceId();
        setLeft( el, newFace );
        assert( isLeftTri( e0 ) );
        assert( left( e0 ) == newFace );
        if ( region && region->test( l ) )
            region->autoResizeSet( newFace );
        setNewToOld( new2Old, {newFace}, l );
    }
    if ( r.valid() && ePrev != e )
    {
        EdgeId er = makeEdge();
        splice( e0.sym(), er );
        splice( prev( ePrev.sym() ), er.sym() );
        auto newFace = addFaceId();
        setLeft( er.sym(), newFace );
        assert( isLeftTri( e0.sym() ) );
        assert( left( e0.sym() ) == newFace );
        if ( region && region->test( r ) )
            region->autoResizeSet( newFace );
        setNewToOld( new2Old, {newFace}, r );
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

VertId MeshTopology::splitFace( FaceId f, FaceBitSet * region, FaceHashMap * new2Old )
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

    setNewToOld( new2Old, { f1, f2 }, f );

    return newv;
}

void MeshTopology::flipOrientation( const UndirectedEdgeBitSet * fullComponents )
{
    MR_TIMER;

    ParallelFor( edgePerFace_, [&]( FaceId f )
    {
        auto e = edgePerFace_[f];
        if ( e && contains( fullComponents, e.undirected() ) )
            edgePerFace_[f] = e.sym();
    } );

    ParallelFor( 0_ue, UndirectedEdgeId( undirectedEdgeSize() ), [&]( UndirectedEdgeId ue )
    {
        if ( fullComponents && !fullComponents->test( ue ) )
            return;
        EdgeId i = ue;
        auto & r0 = edges_[i];
        std::swap( r0.next, r0.prev );

        auto & r1 = edges_[i + 1];
        std::swap( r1.next, r1.prev );

        std::swap( r0.left, r1.left );
    } );
}

void MeshTopology::addPart( const MeshTopology & from,
    FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap, bool rearrangeTriangles )
{
    addPart( from, Src2TgtMaps( outFmap, outVmap, outEmap ), rearrangeTriangles );
}

void MeshTopology::addPart( const MeshTopology & from, const PartMapping & map, bool rearrangeTriangles )
{
    MR_TIMER;

    if ( !rearrangeTriangles )
    {
        // addPartByMask is better optimized, but does not support triangles' rearrangement

        // historically addPartByMask uses hash-maps by default, and addPart uses dense maps by default;
        // keep this behavior
        auto map1 = map;

        FaceMapOrHashMap fmap;
        if ( !map1.src2tgtFaces )
        {
            fmap = FaceMapOrHashMap::createMap();
            map1.src2tgtFaces = &fmap;
        }

        WholeEdgeMapOrHashMap emap;
        if ( !map1.src2tgtEdges )
        {
            emap = WholeEdgeMapOrHashMap::createMap();
            map1.src2tgtEdges = &emap;
        }

        VertMapOrHashMap vmap;
        if ( !map1.src2tgtVerts )
        {
            vmap = VertMapOrHashMap::createMap();
            map1.src2tgtVerts = &vmap;
        }

        return addPartByMask( from, nullptr, map1 );
    }

    assert( from.updateValids_ );

    // maps: to index -> from index
    if ( map.tgt2srcEdges )
        map.tgt2srcEdges->resizeReserve( undirectedEdgeSize(), from.undirectedEdgeSize() );
    if ( map.tgt2srcVerts )
        map.tgt2srcVerts->resizeReserve( vertSize(), from.numValidVerts() );
    if ( map.tgt2srcFaces )
        map.tgt2srcFaces->resizeReserve( faceSize(), from.numValidFaces() );

    // (f/e/v)maps: from index -> to index;
    // use hash map only if requested by the user, otherwise dense map

    auto emap = map.src2tgtEdges ? std::move( *map.src2tgtEdges ) : WholeEdgeMapOrHashMap::createMap();
    const auto ueSize = from.undirectedEdgeSize();
    emap.resizeReserve( ueSize, ueSize );
    EdgeId firstNewEdge = edges_.endId();
    for ( UndirectedEdgeId i{ 0 }; i < ueSize; ++i )
    {
        if ( from.isLoneEdge( i ) )
            continue;
        setAt( emap, i, edges_.endId() );
        if ( map.tgt2srcEdges )
            map.tgt2srcEdges->pushBack( UndirectedEdgeId{ undirectedEdgeSize() }, EdgeId{ i } );
        edges_.push_back( from.edges_[ EdgeId( i ) ] );
        edges_.push_back( from.edges_[ EdgeId( i ).sym() ] );
    }

    auto vmap = map.src2tgtVerts ? std::move( *map.src2tgtVerts ) : VertMapOrHashMap::createMap();
    const auto vSize = from.vertSize();
    vmap.resizeReserve( vSize, from.numValidVerts() );
    for ( VertId i{ 0 }; i < vSize; ++i )
    {
        auto efrom = from.edgePerVertex_[i];
        if ( !efrom.valid() )
            continue;
        auto nv = addVertId();
        setAt( vmap, i, nv );
        if ( map.tgt2srcVerts )
            map.tgt2srcVerts->pushBack( nv, i );
        edgePerVertex_[nv] = mapEdge( emap, efrom );
        if ( updateValids_ )
        {
            validVerts_.set( nv );
            ++numValidVerts_;
        }
    }

    auto fmap = map.src2tgtFaces ? std::move( *map.src2tgtFaces ) : FaceMapOrHashMap::createMap();
    const auto fSize = from.faceSize();
    fmap.resizeReserve( fSize, from.numValidFaces() );
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

        tbb::parallel_sort( begin( invMap ), end( invMap ), isFromFaceLess );
        for ( auto i : invMap )
        {
            auto nf = addFaceId();
            setAt( fmap, i, nf );
            if ( map.tgt2srcFaces )
                map.tgt2srcFaces->pushBack( nf, i );
        }
    }
    else
    {
        for ( auto i : from.validFaces_ )
        {
            auto nf = addFaceId();
            setAt( fmap, i, nf );
            if ( map.tgt2srcFaces )
                map.tgt2srcFaces->pushBack( nf, i );
        }
    }

    for ( FaceId i{ 0 }; i < fSize; ++i )
    {
        auto efrom = from.edgePerFace_[i];
        if ( !efrom.valid() )
            continue;
        auto nf = getAt( fmap, i );
        edgePerFace_[nf] = mapEdge( emap, efrom );
    }
    if ( updateValids_ )
    {
        validFaces_.set( firstNewFace, from.numValidFaces_, true );
        numValidFaces_ += from.numValidFaces_;
    }

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

#ifndef NDEBUG
    if ( map.tgt2srcEdges )
        if ( auto m = map.tgt2srcEdges->getMap() )
            assert( m->size() == undirectedEdgeSize() );
    if ( map.tgt2srcVerts )
        if ( auto m = map.tgt2srcVerts->getMap() )
            assert( m->size() == vertSize() );
    if ( map.tgt2srcFaces )
        if ( auto m = map.tgt2srcFaces->getMap() )
            assert( m->size() == faceSize() );
#endif

    if ( map.src2tgtFaces )
        *map.src2tgtFaces = std::move( fmap );
    if ( map.src2tgtVerts )
        *map.src2tgtVerts = std::move( vmap );
    if ( map.src2tgtEdges )
        *map.src2tgtEdges = std::move( emap );
}

bool MeshTopology::operator ==( const MeshTopology & b ) const
{
    MR_TIMER;
    // make fast comparisons first
    if ( updateValids_ && b.updateValids_ )
    {
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
    }

    return edges_ == b.edges_;
}

void MeshTopology::resizeBeforeParallelAdd( size_t edgeSize, size_t vertSize, size_t faceSize )
{
    MR_TIMER;

    updateValids_ = false;

    edges_.resizeNoInit( edgeSize );

    edgePerVertex_.resize( vertSize );
    validVerts_.resize( vertSize );

    edgePerFace_.resize( faceSize );
    validFaces_.resize( faceSize );
}

void MeshTopology::addPackedPart( const MeshTopology & from, EdgeId toEdgeId, const FaceMap & fmap, const VertMap & vmap )
{
    MR_TIMER;

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

void MeshTopology::stopUpdatingValids()
{
    assert( updateValids_ );
    updateValids_ = false;
#ifndef NDEBUG
    validFaces_ = {};
    validVerts_ = {};
    numValidFaces_ = -1;
    numValidFaces_ = -1;
#endif
}

void MeshTopology::preferEdges( const UndirectedEdgeBitSet & stableEdges )
{
    MR_TIMER;

    tbb::parallel_for( tbb::blocked_range( 0_f, edgePerFace_.endId() ), [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
            for ( EdgeId e : leftRing( *this, f ) )
                if ( stableEdges.test( e.undirected() ) )
                {
                    edgePerFace_[f] = e;
                    break;
                }
    } );

    tbb::parallel_for( tbb::blocked_range( 0_v, edgePerVertex_.endId() ), [&]( const tbb::blocked_range<VertId> & range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
            for ( EdgeId e : orgRing( *this, v ) )
                if ( stableEdges.test( e.undirected() ) )
                {
                    edgePerVertex_[v] = e;
                    break;
                }
    } );
}

bool MeshTopology::buildGridMesh( const GridSettings & settings, ProgressCallback cb )
{
    MR_TIMER;

    stopUpdatingValids();

    // we use resizeNoInit because expect vertices/faces/edges to be tightly packed (no deleted elements within valid range)
    // note: some vertices might be valid but have no edge
    edgePerVertex_.resizeNoInit( settings.vertIds.tsize );
    edgePerFace_.resizeNoInit( settings.faceIds.tsize );
    edges_.resizeNoInit( 2 * settings.uedgeIds.tsize );

    auto getVertId = [&]( Vector2i v ) -> VertId
    {
        if ( v.x < 0 || v.x > settings.dim.x || v.y < 0 || v.y > settings.dim.y )
            return VertId();
        return settings.vertIds.b[ v.x + v.y * ( settings.dim.x + 1 ) ];
    };
    auto getFaceId = [&]( Vector2i v, GridSettings::TriType triType ) -> FaceId
    {
        if ( v.x < 0 || v.x >= settings.dim.x || v.y < 0 || v.y >= settings.dim.y )
            return FaceId();
        return settings.faceIds.b[ 2 * ( v.x + v.y * settings.dim.x ) + (int)triType ];
    };
    auto getEdgeId = [&]( Vector2i v, GridSettings::EdgeType edgeType ) -> EdgeId
    {
        if ( v.x < 0 || v.x > settings.dim.x || v.y < 0 || v.y > settings.dim.y )
            return EdgeId();
        auto ue = settings.uedgeIds.b[ 4 * ( v.x + v.y * ( settings.dim.x + 1 ) ) + (int)edgeType ];
        return ue ? EdgeId( ue ) : EdgeId();
    };

    struct EdgeFace
    {
        EdgeId e;
        FaceId f; //to the left of e
    };
    tbb::enumerable_thread_specific<std::vector<EdgeFace>> edgeRingPerThread;
    auto result = ParallelFor( 0, settings.dim.y + 1, edgeRingPerThread, [&]( int y, std::vector<EdgeFace> & edgeRing )
    {
        Vector2i pos;
        pos.y = y;
        for ( pos.x = 0; pos.x <= settings.dim.x; ++pos.x )
        {
            if ( auto da = getEdgeId( pos, GridSettings::EdgeType::DiagonalA ) )
            {
                if ( const auto fl = getFaceId( pos, GridSettings::TriType::Lower ) )
                    edgePerFace_[fl] = da.sym();
                if ( const auto fu = getFaceId( pos, GridSettings::TriType::Upper ) )
                    edgePerFace_[fu] = da;
            }
            else if ( auto db = getEdgeId( pos, GridSettings::EdgeType::DiagonalB ) )
            {
                if ( const auto fl = getFaceId( pos, GridSettings::TriType::Lower ) )
                    edgePerFace_[fl] = db;
                if ( const auto fu = getFaceId( pos, GridSettings::TriType::Upper ) )
                    edgePerFace_[fu] = db.sym();
            }
            const auto v = getVertId( pos );
            if ( !v )
                continue;
            edgeRing.clear();

            // edge (+1, 0)
            if ( auto e = getEdgeId( pos, GridSettings::EdgeType::Horizontal ) )
                edgeRing.push_back( { e, getFaceId( pos, GridSettings::TriType::Lower ) } );

            // edge (+1, +1)
            if ( auto e = getEdgeId( pos, GridSettings::EdgeType::DiagonalA ) )
                edgeRing.push_back( { e, getFaceId( pos, GridSettings::TriType::Upper ) } );

            // edge (0, +1)
            if ( auto e = getEdgeId( pos, GridSettings::EdgeType::Vertical ) )
            {
                if ( getEdgeId( pos - Vector2i(1, 0), GridSettings::EdgeType::DiagonalA ) )
                    edgeRing.push_back( { e, getFaceId( pos - Vector2i(1, 0), GridSettings::TriType::Lower ) } );
                else if ( getEdgeId( pos - Vector2i(1, 0), GridSettings::EdgeType::DiagonalB ) )
                    edgeRing.push_back( { e, getFaceId( pos - Vector2i(1, 0), GridSettings::TriType::Upper ) } );
                else
                    edgeRing.push_back( { e, FaceId{} } );
            }

            // edge (-1, +1)
            if ( auto e = getEdgeId( pos - Vector2i(1, 0), GridSettings::EdgeType::DiagonalB ) )
                edgeRing.push_back( { e, getFaceId( pos - Vector2i(1, 0), GridSettings::TriType::Lower ) } );

            // edge (-1, 0)
            if ( auto e = getEdgeId( pos - Vector2i(1, 0), GridSettings::EdgeType::Horizontal ) )
                edgeRing.push_back( { e.sym(), getFaceId( pos - Vector2i(1, 1), GridSettings::TriType::Upper ) } );

            // edge (-1, -1)
            if ( auto e = getEdgeId( pos - Vector2i(1, 1), GridSettings::EdgeType::DiagonalA ) )
                edgeRing.push_back( { e.sym(), getFaceId( pos - Vector2i(1, 1), GridSettings::TriType::Lower ) } );

            // edge (0, -1)
            if ( auto e = getEdgeId( pos - Vector2i(0, 1), GridSettings::EdgeType::Vertical ) )
            {
                if ( getEdgeId( pos - Vector2i(0, 1), GridSettings::EdgeType::DiagonalA ) )
                    edgeRing.push_back( { e.sym(), getFaceId( pos - Vector2i(0, 1), GridSettings::TriType::Upper ) } );
                else if ( getEdgeId( pos - Vector2i(0, 1), GridSettings::EdgeType::DiagonalB ) )
                    edgeRing.push_back( { e.sym(), getFaceId( pos - Vector2i(0, 1), GridSettings::TriType::Lower ) } );
                else
                    edgeRing.push_back( { e.sym(), FaceId{} } );
            }

            // edge (+1, -1)
            if ( auto e = getEdgeId( pos - Vector2i(0, 1), GridSettings::EdgeType::DiagonalB ) )
                edgeRing.push_back( { e.sym(), getFaceId( pos - Vector2i(0, 1), GridSettings::TriType::Upper ) } );

            if ( edgeRing.empty() )
            {
                // grid has valid vertex with no connections
                // init edgePerVertex_[v] with invalid edge to override garbage from resizeNoInit
                // (this is only possible case of unpacked vertices here)
                edgePerVertex_[v] = {};
                continue;
            }
            edgePerVertex_[v] = edgeRing[0].e;
            for ( int i = 0; i < edgeRing.size(); ++i )
            {
                HalfEdgeRecord he( noInit );
                he.next = i + 1 < edgeRing.size() ? edgeRing[i + 1].e : edgeRing[0].e;
                he.prev = i > 0 ? edgeRing[i - 1].e : edgeRing.back().e;
                he.org = v;
                he.left = edgeRing[i].f;
                edges_[edgeRing[i].e] = he;
            }
        }
    }, subprogress( cb, 0.0f, 0.5f ) );

    if ( !result )
        return result;

    return computeValidsFromEdges( subprogress( cb, 0.5f, 1.0f ) );
}

bool MeshTopology::computeValidsFromEdges( ProgressCallback cb )
{
    MR_TIMER;
    assert( !updateValids_ );

    validVerts_.clear();
    validVerts_.resize( edgePerVertex_.size() );
    auto result = BitSetParallelForAll( validVerts_, [&]( VertId v )
    {
        if ( edgePerVertex_[v].valid() )
            validVerts_.set( v );
    }, subprogress( cb, 0.0f, 0.3f ) );

    if ( !result )
        return result;

    numValidVerts_ = parallel_reduce( tbb::blocked_range( 0_v, VertId{ vertSize() } ), 0,
    [&] ( const auto & range, int curr )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
            if ( validVerts_.test( v ) )
                ++curr;
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );

    validFaces_.clear();
    validFaces_.resize( edgePerFace_.size() );
    result = BitSetParallelForAll( validFaces_, [&]( FaceId f )
    {
        if ( edgePerFace_[f].valid() )
            validFaces_.set( f );
    }, subprogress( cb, 0.6f, 0.9f ) );

    if ( !result )
        return result;

    numValidFaces_ = parallel_reduce( tbb::blocked_range( 0_f, FaceId{ faceSize() } ), 0,
    [&] ( const auto & range, int curr )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
            if ( validFaces_.test( f ) )
                ++curr;
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );

    updateValids_ = true;

    if ( cb && !cb( 1.0f ) )
    {
        return false;
    }

    return true;
}

void MeshTopology::computeAllFromEdges_()
{
    MR_TIMER;

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

void MeshTopology::addPartByMask( const MeshTopology & from, const FaceBitSet * fromFaces, const PartMapping & map )
{
    addPartByMask( from, fromFaces, false, {}, {}, map );
}

void MeshTopology::addPartByMask( const MeshTopology & from, const FaceBitSet * fromFaces0, bool flipOrientation,
    const std::vector<EdgePath> & thisContours,
    const std::vector<EdgePath> & fromContours,
    const PartMapping & map )
{
    MR_TIMER;
    const auto szContours = thisContours.size();
    assert( szContours == fromContours.size() );

    const auto & fromFaces = from.getFaceIds( fromFaces0 );
    const auto fcount = fromFaces.count();

    // maps: from index -> to index;
    // use dense map only if requested by the user, otherwise hash map
    auto fmap = map.src2tgtFaces ? std::move( *map.src2tgtFaces ) : FaceMapOrHashMap::createHashMap();
    fmap.resizeReserve( from.faceSize(), fcount );

    auto emap = map.src2tgtEdges ? std::move( *map.src2tgtEdges ) : WholeEdgeMapOrHashMap::createHashMap();
    emap.resizeReserve( from.undirectedEdgeSize(), std::min( 2 * fcount, from.undirectedEdgeSize() ) ); // if whole connected component is copied then ecount=3/2*fcount; if unconnected triangles are copied then ecount=3*fcount

    auto vmap = map.src2tgtVerts ? std::move( *map.src2tgtVerts ) : VertMapOrHashMap::createHashMap();
    vmap.resizeReserve( from.vertSize(), std::min( fcount, from.vertSize() ) ); // if whole connected component is copied then vcount=1/2*fcount; if unconnected triangles are copied then vcount=3*fcount

    // maps: to index -> from index
    if ( map.tgt2srcEdges )
        map.tgt2srcEdges->resizeReserve( undirectedEdgeSize(), std::min( 2 * fcount, from.undirectedEdgeSize() ) );
    if ( map.tgt2srcVerts )
        map.tgt2srcVerts->resizeReserve( vertSize(), std::min( fcount, from.vertSize() ) );
    if ( map.tgt2srcFaces )
        map.tgt2srcFaces->resizeReserve( faceSize(), fcount );

    VertBitSet fromMappedVerts( from.vertSize() );
    auto setVmap = [&] ( VertId key, VertId val )
    {
        if ( !fromMappedVerts.test_set( key ) )
        {
            assert( !getAt( vmap, key ) );
            setAt( vmap, key, val );
        }
#ifndef NDEBUG
        else
        {
            assert( getAt( vmap, key ) == val );
        }
#endif
    };

    UndirectedEdgeBitSet fromMappedEdges( from.undirectedEdgeSize() ); //one of fromContours' edge
    for ( int i = 0; i < szContours; ++i )
    {
        const auto & thisContour = thisContours[i];
        const auto & fromContour = fromContours[i];
        const auto sz = thisContour.size();
        assert( sz == fromContour.size() );
        if ( thisContour.empty() )
            continue;
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
            setVmap( from.org( e ), org( e1 ) );
            setVmap( from.dest( e ), dest( e1 ) );
            assert( !getAt( emap, e.undirected() ) );
            setAt( emap, e.undirected(), e.even() ? e1 : e1.sym() );
            fromMappedEdges.set( e.undirected() );
        }
    }

    const EdgeId firstNewEdge = edges_.endId();
    EdgeId nextNewEdge = firstNewEdge;
    auto copyEdge = [&]( UndirectedEdgeId fromUe )
    {
        assert( !getAt( emap, fromUe ) );
        setAt( emap, fromUe, nextNewEdge );
        if ( map.tgt2srcEdges )
            map.tgt2srcEdges->pushBack( UndirectedEdgeId{ nextNewEdge }, EdgeId{ fromUe } );
        nextNewEdge += 2;
    };

    const VertId firstNewVert = edgePerVertex_.endId();
    VertId nextNewVert = firstNewVert;
    auto copyVert = [&]( VertId fromV )
    {
        auto nv = nextNewVert++;
        assert( !getAt( vmap, fromV ) );
        setAt( vmap, fromV, nv );
        if ( map.tgt2srcVerts )
            map.tgt2srcVerts->pushBack( nv, fromV );
    };

    const FaceId firstNewFace = edgePerFace_.endId();
    FaceId nextNewFace = firstNewFace;
    auto copyFace = [&]( FaceId fromF )
    {
        auto nf = nextNewFace++;
        if ( map.tgt2srcFaces )
            map.tgt2srcFaces ->pushBack( nf, fromF );
        setAt( fmap, fromF, nf );
    };

    // fill all maps
    VertBitSet fromCopiedVerts; // except for moved vertices
    UndirectedEdgeBitSet fromCopiedEdges; // except for moved edges
    if ( fromFaces0 )
    {
        fromCopiedVerts = fromMappedVerts;
        fromCopiedEdges = fromMappedEdges;
        for ( auto f : fromFaces )
        {
            auto efrom = from.edgePerFace_[f];
            for ( auto e : leftRing( from, efrom ) )
            {
                const UndirectedEdgeId ue = e.undirected();
                if ( !fromCopiedEdges.test_set( ue ) )
                    copyEdge( ue );
                if ( auto v = from.org( e ); v.valid() )
                {
                    if ( !fromCopiedVerts.test_set( v ) )
                        copyVert( v );
                }
            }
            copyFace( f );
        }
        fromCopiedVerts -= fromMappedVerts;
        fromCopiedEdges -= fromMappedEdges;
    }
    else
    {
        // whole (from) mesh is copied
        tbb::task_group taskGroup;
        taskGroup.run( [&] ()
        {
            fromCopiedVerts = from.getValidVerts() - fromMappedVerts;
            for ( auto v : fromCopiedVerts )
                copyVert( v );
        } );

        taskGroup.run( [&] ()
        {
            for ( auto f : from.getValidFaces() )
                copyFace( f );
        } );

        fromCopiedEdges = from.findNotLoneUndirectedEdges() - fromMappedEdges;
        for ( auto ue : fromCopiedEdges )
            copyEdge( ue );

        taskGroup.wait();
    }

    // in case of open contours, some nearby edges have to be updated
    std::vector<EdgePair> prevNextEdges;
    for ( int i = 0; i < szContours; ++i )
    {
        const auto & thisContour = thisContours[i];
        const auto & fromContour = fromContours[i];
        const auto sz = thisContour.size();
        for ( int j = 0; j < sz; ++j )
        {
            auto e = fromContour[j];
            auto e1 = thisContour[j];

            EdgeId eNx = e.sym();
            for ( ;;) // loop is needed in case of some ring part is not in fromPart, so find next edge in fromPart
            {
                eNx = flipOrientation ? from.prev( eNx ) : from.next( eNx );
                auto cf = flipOrientation ? from.right( eNx ) : from.left( eNx );
                if ( getAt( fmap, cf ) || eNx == e.sym() )
                    break;
            }
            if ( !fromMappedEdges.test( eNx.undirected() ) )
            {
                auto e1Nx = prev( e1.sym() );
                prevNextEdges.emplace_back( e1Nx, mapEdge( emap, eNx ) );
            }

            EdgeId ePr = e;
            for ( ;;) // loop is needed in case of some ring part is not in fromPart, so find prev edge in fromPart
            {
                ePr = flipOrientation ? from.next( ePr ) : from.prev( ePr );
                auto cf = flipOrientation ? from.left( ePr ) : from.right( ePr );
                if ( getAt( fmap, cf ) || ePr == e )
                    break;
            }
            if ( !fromMappedEdges.test( ePr.undirected() ) )
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
                assert ( fromMappedEdges.test( e.undirected() ) );
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

    // translate edge records
    edges_.resizeNoInit( nextNewEdge );
    BitSetParallelFor( fromCopiedEdges, [&]( UndirectedEdgeId fromUe )
    {
        auto e0 = from.edges_[EdgeId{ fromUe }];
        auto e1 = from.edges_[EdgeId{ fromUe }.sym()];
        from.translate_( e0, e1, fmap, vmap, emap, flipOrientation );

        const UndirectedEdgeId nue = getAt( emap, fromUe );
        const EdgeId ne{ nue };
        edges_[ne] = e0;
        edges_[ne.sym()] = e1;
    } );

    // translate vertex records
    if ( updateValids_ )
    {
        validVerts_.autoResizeSet( firstNewVert, nextNewVert - firstNewVert, true );
        numValidVerts_ += nextNewVert - firstNewVert;
    }
    edgePerVertex_.resizeNoInit( nextNewVert );
    BitSetParallelFor( fromCopiedVerts, [&]( VertId v )
    {
        for ( auto fromE : orgRing( from, v ) )
            if ( auto e = mapEdge( emap, fromE ) )
            {
                edgePerVertex_[getAt( vmap, v )] = e;
                return;
            }
        assert( !"at least one edge of vertex must be copied" );
    } );

    // translate face records
    if ( updateValids_ )
    {
        validFaces_.autoResizeSet( firstNewFace, nextNewFace - firstNewFace, true );
        numValidFaces_ += nextNewFace - firstNewFace;
    }
    edgePerFace_.resizeNoInit( nextNewFace );
    BitSetParallelFor( fromFaces, [&]( FaceId f )
    {
        for ( auto fromE : leftRing( from, f ) )
            if ( auto e = mapEdge( emap, fromE ) )
            {
                edgePerFace_[getAt( fmap, f )] = flipOrientation ? e.sym() : e;
                return;
            }
        assert( !"at least one edge of face must be copied" );
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

#ifndef NDEBUG
    if ( map.tgt2srcEdges )
        if ( auto m = map.tgt2srcEdges->getMap() )
            assert( m->size() == undirectedEdgeSize() );
    if ( map.tgt2srcVerts )
        if ( auto m = map.tgt2srcVerts->getMap() )
            assert( m->size() == vertSize() );
    if ( map.tgt2srcFaces )
        if ( auto m = map.tgt2srcFaces->getMap() )
            assert( m->size() == faceSize() );
#endif

    if ( map.src2tgtFaces )
        *map.src2tgtFaces = std::move( fmap );
    if ( map.src2tgtVerts )
        *map.src2tgtVerts = std::move( vmap );
    if ( map.src2tgtEdges )
        *map.src2tgtEdges = std::move( emap );
}

void MeshTopology::rotateTriangles()
{
    MR_TIMER;

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
    MR_TIMER;

    if ( rearrangeTriangles )
        rotateTriangles();
    MeshTopology packed;
    packed.vertReserve( numValidVerts() );
    packed.faceReserve( numValidFaces() );
    packed.edgeReserve( 2 * computeNotLoneUndirectedEdges() );
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
    MR_TIMER;

    Vector<NoDefInit<HalfEdgeRecord>, UndirectedEdgeId> tmp( map.e.tsize );
    auto translateHalfEdge = [&]( const HalfEdgeRecord & he )
    {
        HalfEdgeRecord res;
        res.next = getAt( map.e.b, he.next );
        res.prev = getAt( map.e.b, he.prev );
        res.org = getAt( map.v.b, he.org );
        res.left = getAt( map.f.b, he.left );
        return res;
    };

    // translate even half-edges
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( undirectedEdgeSize() ) ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( auto oldUe = range.begin(); oldUe < range.end(); ++oldUe )
        {
            auto newUe = map.e.b[oldUe];
            if ( !newUe )
                continue;
            tmp[ newUe ] = translateHalfEdge( edges_[ EdgeId{oldUe} ] );
        }
    } );
    // copy back even half-edges
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( map.e.tsize ) ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( auto newUe = range.begin(); newUe < range.end(); ++newUe )
        {
            edges_[ EdgeId{newUe} ] = tmp[ newUe ];
        }
    } );

    // translate odd half-edges
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( undirectedEdgeSize() ) ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( auto oldUe = range.begin(); oldUe < range.end(); ++oldUe )
        {
            auto newUe = map.e.b[oldUe];
            if ( !newUe )
                continue;
            tmp[ newUe ] = translateHalfEdge( edges_[ EdgeId{oldUe}.sym() ] );
        }
    } );
    // copy back odd half-edges
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( map.e.tsize ) ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( auto newUe = range.begin(); newUe < range.end(); ++newUe )
        {
            edges_[ EdgeId{newUe}.sym() ] = tmp[ newUe ];
        }
    } );

    tmp = {};
    edges_.resize( 2 * map.e.tsize );

    Vector<EdgeId, FaceId> newEdgePerFace;
    newEdgePerFace.resizeNoInit( map.f.tsize );
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
    validFaces_.clear();
    validFaces_.resize( edgePerFace_.size(), true );

    Vector<EdgeId, VertId> newEdgePerVertex;
    newEdgePerVertex.resizeNoInit( map.v.tsize );
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
    validVerts_.clear();
    validVerts_.resize( edgePerVertex_.size(), true );
    updateValids_ = true;
}

/// reorders elements given \ref map: old -> new, a getter \ref get and a setter \ref put
template<typename T, typename G, typename P>
static void shuffle( const BMap<Id<T>, Id<T>> & map, G && get, P && put )
{
    MR_TIMER;

    TaggedBitSet<T> replacedByNew( map.tsize );
    for ( Id<T> i{0}; i < map.b.size(); ++i )
    {
        if ( replacedByNew.test( i ) )
            continue;
        auto j = map.b[i];
        if ( !j || i == j )
            continue;
        if ( j < i )
        {
            // value at #j has been copied already
            put( j, get( i ) );
            continue;
        }
        auto storedVal = get( j );
        put( j, get( i ) );
        replacedByNew.set( j );
        for ( j = map.b[j]; j > i; j = map.b[j] )
        {
            assert ( !replacedByNew.test( j ) );
            auto tmp = get( j );
            put( j, storedVal );
            replacedByNew.set( j );
            storedVal = tmp;
        }
        if ( j )
            put( j, storedVal );
    }
}

void MeshTopology::packMinMem( const PackMapping & map )
{
    MR_TIMER;
    assert( map.f.tsize == numValidFaces_ );
    assert( map.v.tsize == numValidVerts_ );
    assert( map.e.tsize <= edgeSize() );

    Timer m( "shuffle" );
    tbb::task_group group;

    group.run( [&] ()
    {
        shuffle( map.f,
            [&]( FaceId f ) { return edgePerFace_[f]; },
            [&]( FaceId f, EdgeId val ) { edgePerFace_[f] = val; } );
        edgePerFace_.resize( numValidFaces_ );
    } );

    group.run( [&] ()
    {
        shuffle( map.v,
            [&]( VertId v ) { return edgePerVertex_[v]; },
            [&]( VertId v, EdgeId val ) { edgePerVertex_[v] = val; } );
        edgePerVertex_.resize( numValidVerts_ );
    } );

    group.run( [&] ()
    {
        validFaces_.clear();
        validFaces_.resize( numValidFaces_, true );
    } );

    group.run( [&] ()
    {
        validVerts_.clear();
        validVerts_.resize( numValidVerts_, true );
    } );

    shuffle( map.e,
        [&]( UndirectedEdgeId ue ) { return std::make_pair( edges_[ EdgeId{ue} ], edges_[ EdgeId{ue}.sym() ] ); },
        [&]( UndirectedEdgeId ue, const auto & val ) { edges_[ EdgeId{ue} ] = val.first; edges_[ EdgeId{ue}.sym() ] = val.second; } );
    edges_.resize( 2 * map.e.tsize );

    group.wait();

    m.restart( "translate" );
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( map.e.tsize ) ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( auto ue = range.begin(); ue < range.end(); ++ue )
        {
            auto translateHalfEdge = [&]( HalfEdgeRecord & he )
            {
                he.next = getAt( map.e.b, he.next );
                he.prev = getAt( map.e.b, he.prev );
                he.org = getAt( map.v.b, he.org );
                he.left = getAt( map.f.b, he.left );
            };
            translateHalfEdge( edges_[ EdgeId{ue} ] );
            translateHalfEdge( edges_[ EdgeId{ue}.sym() ] );
        }
    } );

    tbb::parallel_for( tbb::blocked_range( 0_f, FaceId( map.f.tsize ) ),
        [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( auto f = range.begin(); f < range.end(); ++f )
        {
            edgePerFace_[f] = getAt( map.e.b, edgePerFace_[f] );
        }
    } );

    tbb::parallel_for( tbb::blocked_range( 0_v, VertId( map.v.tsize ) ),
        [&]( const tbb::blocked_range<VertId> & range )
    {
        for ( auto v = range.begin(); v < range.end(); ++v )
        {
            edgePerVertex_[v] = getAt( map.e.b, edgePerVertex_[v] );
        }
    } );

    updateValids_ = true;
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

Expected<void> MeshTopology::read( std::istream & s, ProgressCallback callback )
{
    updateValids_ = false;

    // read edges
    std::uint32_t numEdges;
    s.read( (char*)&numEdges, 4 );
    if ( !s )
        return unexpected( std::string( "Stream reading error" ) );

    const auto streamSize = getStreamSize( s );
    if ( size_t( streamSize ) < numEdges * sizeof( HalfEdgeRecord ) )
        return unexpected( std::string( "Stream reading error: stream is too short" ) ); // stream is too short

    edges_.resize( numEdges );
    if ( !readByBlocks( s, ( char* )edges_.data(), edges_.size() * sizeof( HalfEdgeRecord ),
        callback ? [callback] ( float v )
    {
        return callback( v / 3.f );
    } : callback ) )
        return unexpectedOperationCanceled();

    // read verts
    std::uint32_t numVerts;
    s.read( (char*)&numVerts, 4 );
    if ( !s )
        return unexpected( std::string( "Stream reading error" ) );
    edgePerVertex_.resize( numVerts );
    if ( !readByBlocks( s, (char*)edgePerVertex_.data(), edgePerVertex_.size() * sizeof( EdgeId ),
        callback ? [callback] ( float v )
    {
        return callback( ( 1.f + v ) / 3.f );
    } : callback ) )
        return unexpectedOperationCanceled();

    // read faces
    std::uint32_t numFaces;
    s.read( (char*)&numFaces, 4 );
    if ( !s )
        return unexpected( std::string( "Stream reading error" ) );
    edgePerFace_.resize( numFaces );
    if ( !readByBlocks( s, (char*)edgePerFace_.data(), edgePerFace_.size() * sizeof( EdgeId ),
        callback ? [callback] ( float v )
    {
        return callback( ( 2.f + v ) / 3.f );
    } : callback ) )
        return unexpectedOperationCanceled();

    computeValidsFromEdges();

    if ( !s.good() )
        return unexpected( std::string( "Stream reading error" ) );
    if ( !checkValidity() )
        return unexpected( std::string( "Data is invalid" ) );
    return {};
}


bool MeshTopology::checkValidity( ProgressCallback cb, bool allVerts ) const
{
    MR_TIMER;

    #define CHECK(x) { assert(x); if (!(x)) return false; }
    CHECK( updateValids_ );
    const auto vSize = edgePerVertex_.size();
    CHECK( vSize == validVerts_.size() )
    const auto fSize = edgePerFace_.size();
    CHECK( fSize == validFaces_.size() )

    std::atomic<bool> failed{ false };
    const auto parCheck = [&]( bool b )
    {
        if ( !b )
            failed.store( true, std::memory_order_relaxed );
    };

    auto result = ParallelFor( edges_, [&] (const EdgeId& e)
    {
        if ( failed.load( std::memory_order_relaxed ) )
            return;
        parCheck( edges_[edges_[e].next].prev == e );
        parCheck( edges_[edges_[e].prev].next == e );
        auto v = edges_[e].org;
        if ( allVerts && !isLoneEdge( e ) )
            parCheck( v.valid() );
        if ( v )
        {
            parCheck( validVerts_.test( v ) );
            // check that vertex v is manifold - there is only one ring of edges around it
            parCheck( edgePerVertex_[v] && fromSameOriginRing( edgePerVertex_[v], e ) );
        }
        if ( auto f = edges_[e].left )
        {
            parCheck( validFaces_.test( f ) );
            // check that face f is manifold - there is only one ring of edges around it
            parCheck( edgePerFace_[f] && fromSameLeftRing( edgePerFace_[f], e ) );
        }
    }, subprogress( cb, 0.0f, 0.3f ) );

    if ( !result )
        return false;
    CHECK( !failed );

    std::atomic<int> realValidVerts{ 0 };

    result = ParallelFor( edgePerVertex_, [&] (const VertId& v)
    {
        int myValidVerts = 0;

        if ( failed.load( std::memory_order_relaxed ) )
            return;
        if ( edgePerVertex_[v].valid() )
        {
            parCheck( validVerts_.test( v ) );
            parCheck( edgePerVertex_[v] < edges_.size() );
            parCheck( edges_[edgePerVertex_[v]].org == v );
            ++myValidVerts;
            for ( EdgeId e : orgRing( *this, v ) )
                parCheck( org(e) == v );
        }
        else
        {
            parCheck( !validVerts_.test( v ) );
        }
        realValidVerts.fetch_add( myValidVerts, std::memory_order_relaxed );
    }, subprogress( cb, 0.3f, 0.6f ) );

    if ( !result )
        return false;

    CHECK( !failed );
    CHECK( numValidVerts_ == realValidVerts );

    std::atomic<int> realValidFaces{ 0 };

    result = ParallelFor( edgePerFace_, [&] (const FaceId& f)
    {
        int myValidFaces = 0;

        if ( failed.load( std::memory_order_relaxed ) )
            return;
        if ( edgePerFace_[f].valid() )
        {
            parCheck( validFaces_.test( f ) );
            parCheck( edgePerFace_[f] < edges_.size() );
            parCheck( edges_[edgePerFace_[f]].left == f );
            ++myValidFaces;
            for ( EdgeId e : leftRing( *this, f ) )
                parCheck( left(e) == f );
        }
        else
        {
            parCheck( !validFaces_.test( f ) );
        }
        realValidFaces.fetch_add( myValidFaces, std::memory_order_relaxed );
    }, subprogress( cb, 0.6f, 1.0f ) );

    if ( !result )
        return false;

    CHECK( !failed );
    CHECK( numValidFaces_ == realValidFaces );

    return true;
}

void loadMeshDll()
{
}

} //namespace MR
