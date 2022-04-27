#include "MRMeshBuilder.h"
#include "MRMeshDelete.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRMesh/MRGTest.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace MeshBuilder
{

class MaxIdCalc 
{
public:
    FaceId maxFaceId;
    VertId maxVertId;

    MaxIdCalc( const std::vector<Triangle> & tris ) : tris_( tris ) { }
    MaxIdCalc( MaxIdCalc & x, tbb::split ) : tris_( x.tris_ ) { }
    void join( const MaxIdCalc & y ) 
    { 
        maxFaceId = std::max( maxFaceId, y.maxFaceId );
        maxVertId = std::max( maxVertId, y.maxVertId );
    }

    void operator()( const tbb::blocked_range<size_t> & r ) 
    {
        for ( size_t i = r.begin(); i < r.end(); ++i ) 
        {
            const auto & t = tris_[i];
            maxFaceId = std::max( maxFaceId, t.f );
            maxVertId = std::max( { maxVertId, t.v[0], t.v[1], t.v[2] } );
        }
    }
            
public:
    const std::vector<Triangle> & tris_;
};

static std::pair<FaceId, VertId> computeMaxIds( const std::vector<Triangle> & tris )
{
    MR_TIMER
    MaxIdCalc calc( tris );
    parallel_reduce( tbb::blocked_range<size_t>( size_t{0}, tris.size() ), calc );
    return { calc.maxFaceId, calc.maxVertId };
}

// returns the edge with the origin in a if it is the only such edge with invalid left face;
// otherwise returns invalid edge
static EdgeId edgeWithOrgAndOnlyLeftHole( const MeshTopology & m, VertId a )
{
    EdgeId e0 = m.edgeWithOrg( a );
    if ( !e0.valid() )
        return {};

    EdgeId eh = e0;
    for (;;)
    {
        if ( !m.left( eh ).valid() )
            break;
        eh = m.next( eh );
        if ( eh == e0 )
            return {}; // no single hole near a
    }

    for ( EdgeId e = m.next( eh ); e != e0; e = m.next( e ) )
    {
        if ( !m.left( e ).valid() )
            return {}; // another hole near a
    }

    return eh;
}

// finds and returns edge from o to d in the mesh that does not have left face; returns invalid edge otherwise
EdgeId findEdgeNoLeft( const MeshTopology & topology, VertId o, VertId d )
{
    assert( o.valid() && d.valid() );
    EdgeId e0 = topology.edgeWithOrg( o );
    if ( !e0.valid() )
        return {};

    for ( EdgeId e = e0;; )
    {
        if ( topology.dest( e ) == d && !topology.left( e ) )
            return e;
        e = topology.next( e );
        if ( e == e0 )
            return {};
    }
}

enum class AddFaceResult
{
    Success,
    // failed to create this face right now, but it may be successful created later when other faces appear in the mesh
    UnsafeTryLater,       // unclear in which order to add new edges to existing vertices
    // permanently failed to add because ...
    FailDegenerateFace,   // ... input face is degenerate
    FailNonManifoldEdge,  // ... one of face edges exists in topology and has another face from same side
    FailNonManifoldVertex // ... one of face vertices will be non-manifold
};

class FaceAdder
{
public:
    AddFaceResult add( MeshTopology& m, FaceId face, const VertId* first, const VertId* last, bool allowNonManifoldEdge = true );

private:
    std::vector<VertId> dupVertices_; //of the face being created
    std::vector<bool> simpleVert_; // vertex does not exist at all or exists as part of our edge
    std::vector<EdgeId> e_; // e[i] - existing edge from v[i] to v[i+1] with no left face
    std::vector<EdgeId> onlyLeftHole_;
};

AddFaceResult FaceAdder::add( MeshTopology & m, FaceId face, const VertId * first, const VertId * last, bool allowNonManifoldEdge )
{
    const auto sz = std::distance( first, last );
    dupVertices_.assign( first, last );
    std::sort( dupVertices_.begin(), dupVertices_.end() );
    if ( adjacent_find( dupVertices_.begin(), dupVertices_.end() ) != dupVertices_.end() )
        return AddFaceResult::FailDegenerateFace; // prohibit creating face with repeating vertices

    simpleVert_.clear();
    simpleVert_.resize( sz, false );

    e_.clear();
    e_.resize( sz );

    onlyLeftHole_.clear();
    onlyLeftHole_.resize( sz );

    // prepare data
    for ( int i = 0; i < sz; ++i )
    {
        if ( !m.hasVert( first[i] ) )
        {
            // 1) vertex does not exist yet - good
            simpleVert_[i] = true;
            continue;
        }

        int i1 = (i + 1) % sz;
        if ( allowNonManifoldEdge )
            e_[i] = findEdgeNoLeft( m, first[i], first[i1] );
        else
            e_[i] = m.findEdge( first[i], first[i1] );
        if ( e_[i].valid() )
        {
            if ( !allowNonManifoldEdge && m.left( e_[i] ).valid() )
                return AddFaceResult::FailNonManifoldEdge; // the edge exists and has another face from the left
            // 2) edge exists but does not have face at this side - good
            simpleVert_[i] = true;
            simpleVert_[i1] = true;
        }
    }

    // for not yet good vertices check that they have one hole nearby
    for ( int i = 0; i < sz; ++i )
    {
        int i2 = (i + (int)sz - 1) % sz;
        if ( e_[i].valid() && e_[i2].valid() )
        {
            if ( m.next( e_[i] ) != e_[i2].sym() )
                return AddFaceResult::FailNonManifoldVertex; // impossible to create non-manifold vertex
        }
        if ( !simpleVert_[i] )
        {
            onlyLeftHole_[i] = edgeWithOrgAndOnlyLeftHole( m, first[i] );
            if ( !onlyLeftHole_[i].valid() )
                return AddFaceResult::UnsafeTryLater; // impossible to safely create a face
        }
    }

    // create missing face edges
    for ( int i = 0; i < sz; ++i )
    {
        if ( !e_[i].valid() )
            e_[i] = m.makeEdge();
    }

    // connect face edges at vertices and set vertex-ids
    for ( int i = 0; i < sz; ++i )
    {
        int i2 = (i + (int)sz - 1) % sz;
        if ( m.org( e_[i] ) == first[i] && m.org( e_[i2].sym() ) == first[i] )
        {
            assert( m.next( e_[i] ) == e_[i2].sym() );
            continue; // both edges already existed and connected at this vertex
        }

        if ( onlyLeftHole_[i].valid() && onlyLeftHole_[i] != e_[i] )
        {
            m.splice( onlyLeftHole_[i], e_[i] );
            assert( m.org( e_[i] ) == first[i] );
        }

        m.splice( e_[i], m.prev( e_[i2].sym() ) );
        assert( m.next( e_[i] ) == e_[i2].sym() );
        m.setOrg( e_[i], first[i] );
    }

    assert( m.getLeftDegree( e_[0] ) == sz );
    m.setLeft( e_[0], face );
    return AddFaceResult::Success;
}

static void addTrianglesSeqCore( MeshTopology& res, std::vector<Triangle>& tris, bool allowNonManifoldEdge = true )
{
    MR_TIMER

    FaceAdder fa;
    // these are triangles remaining from the previous pass to try again
    std::vector<Triangle> nextPass;
    // these are triangles that cannot be added even after other triangles
    std::vector<Triangle> bad;
    while ( !tris.empty() )
    {
        for ( const auto & tri : tris )
        {
            auto x = fa.add( res, tri.f, tri.v, tri.v + 3, allowNonManifoldEdge );
            if ( x == AddFaceResult::UnsafeTryLater )
                nextPass.push_back( tri );
            else if ( x != AddFaceResult::Success )
                bad.push_back( tri );
        }

        if ( nextPass.size() == tris.size() )
            break; // no single triangle added during the pass
        tris.swap( nextPass );
        nextPass.clear();
    }
    tris.insert( tris.end(), bad.begin(), bad.end() );
}

MeshTopology fromFaceSoup( const std::vector<VertId> & verts, std::vector<FaceRecord> & faces )
{
    MR_TIMER

    MeshTopology res;
    if ( faces.empty() || verts.empty() )
        return res;

    // reserve enough elements for faces and vertices
    auto maxFaceId = std::max_element( faces.begin(), faces.end(), 
        []( const FaceRecord & a, const FaceRecord & b ) { return a.face < b.face; } )->face;
    auto maxVertId = *std::max_element( verts.begin(), verts.end() );
    res.faceResize( maxFaceId + 1 );
    res.vertResize( maxVertId + 1 );

    FaceAdder fa;
    // these are faces remaining from the previous pass to try again
    std::vector<FaceRecord> nextPass;
    // these are faces that cannot be added even after other faces
    std::vector<FaceRecord> bad;
    while ( !faces.empty() )
    {
        for ( const auto & faceRec : faces )
        {
            auto x = fa.add( res, faceRec.face, verts.data() + faceRec.firstVertex, verts.data() + faceRec.lastVertex );
            if ( x == AddFaceResult::UnsafeTryLater )
                nextPass.push_back( faceRec );
            else if ( x != AddFaceResult::Success )
                bad.push_back( faceRec );
        }

        if ( nextPass.size() == faces.size() )
            break; // no single face added during the pass
        faces.swap( nextPass );
        nextPass.clear();
    }
    faces.insert( faces.end(), bad.begin(), bad.end() );
    return res;
}

void addTriangles( MeshTopology& res, std::vector<Triangle>& tris, bool allowNonManifoldEdge )
{
    MR_TIMER

    // reserve enough elements for faces and vertices
    auto [maxFaceId, maxVertId] = computeMaxIds( tris );
    res.faceResize( maxFaceId + 1 );
    res.vertResize( maxVertId + 1 );

    addTrianglesSeqCore( res, tris, allowNonManifoldEdge );
}

void addTriangles( MeshTopology & res, std::vector<VertId> & vertTriples,
    FaceBitSet * createdFaces )
{
    MR_TIMER

    const int numTri = (int)vertTriples.size() / 3;
    std::vector<Triangle> tris;
    tris.reserve( numTri );
    auto firstNewFace = res.lastValidFace() + 1;

    for ( int t = 0; t < numTri; ++t )
    {
        Triangle tri
        {
            vertTriples[3*t],
            vertTriples[3*t+1], 
            vertTriples[3*t+2],
            firstNewFace + t
        };
        tris.push_back( tri );
    }

    if ( createdFaces )
    {
        if ( createdFaces->size() <= firstNewFace + numTri )
            createdFaces->resize( firstNewFace + numTri + 1 );
        createdFaces->set( firstNewFace, numTri, true );
    }

    vertTriples.clear();
    addTriangles( res, tris );
    for ( const auto & t : tris )
    {
        vertTriples.push_back( t.v[0] );
        vertTriples.push_back( t.v[1] );
        vertTriples.push_back( t.v[2] );
        if ( createdFaces )
            createdFaces->reset( t.f );
    }
}

static MeshTopology fromTrianglesSeq( std::vector<Triangle> & tris )
{
    MeshTopology res;
    addTriangles( res, tris );
    return res;
}

MeshTopology fromDisjointMeshPieces( const std::vector<MeshPiece> & pieces, VertId maxVertId, FaceId maxFaceId, std::vector<Triangle> & borderTris )
{
    MR_TIMER

    // construct empty final mesh with enough number of elements
    const auto numParts = pieces.size();
    std::vector<EdgeId> firstPartEdge( numParts + 1 );
    firstPartEdge[0] = 0_e;
    for ( int i = 0; i < pieces.size(); ++i )
    {
        const auto & p = pieces[i];
        firstPartEdge[i + 1] = firstPartEdge[i] + (int)p.topology.edgeSize();
        for ( const auto & t : p.tris ) // remaining triangles
        {
            borderTris.emplace_back( p.vmap[t.v[0]], p.vmap[t.v[1]], p.vmap[t.v[2]], p.fmap[t.f] );
        }
    }
    int numEdgesInParts = firstPartEdge.back();

    MeshTopology res;
    res.edgeReserve( numEdgesInParts + 6 * borderTris.size() ); // should be enough even of all border triangles are disconnected
    res.resizeBeforeParallelAdd( numEdgesInParts, maxVertId + 1, maxFaceId + 1 );

    // add pieces in res in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, numParts, 1 ), [&]( const tbb::blocked_range<size_t> & range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t myPartId = range.begin(); myPartId < range.end(); ++myPartId )
        {
            const auto & part = pieces[myPartId];
            res.addPackedPart( part.topology, firstPartEdge[myPartId], part.fmap, part.vmap );
        }
    } );
    res.computeValidsFromEdges();

    // and add border triangles
    addTrianglesSeqCore( res, borderTris );

    return res;
}

MeshTopology fromTriangles( const std::vector<Triangle> & tris, std::vector<Triangle> * skippedTris )
{
    if ( tris.empty() )
        return {};
    MR_TIMER
    
    // reserve enough elements for faces and vertices
    auto [maxFaceId, maxVertId] = computeMaxIds( tris );

    constexpr size_t minTrisInPart = 32768;
    // numParts shall not depend on hardware (e.g. on std::thread::hardware_concurrency()) to be repeatable on all hardware
    const size_t numParts = std::min( ( tris.size() + minTrisInPart - 1 ) / minTrisInPart, (size_t)64 );
    assert( numParts >= 1 );
    MeshTopology res;
    if ( numParts <= 1 )
    {
        auto trisCopy = tris;
        res = fromTrianglesSeq( trisCopy );
        if ( skippedTris )
            *skippedTris = std::move( trisCopy );
        return res;
    }

    const size_t vertsInPart = ( (int)maxVertId + numParts ) / numParts;
    std::vector<MeshPiece> parts( numParts );
    std::vector<Triangle> borderTris; // triangles having vertices in distinct parts, computed by thread 0

    Timer timer("parallel parts");
    // construct mesh parts
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, numParts, 1 ), [&]( const tbb::blocked_range<size_t> & range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t myPartId = range.begin(); myPartId < range.end(); ++myPartId )
        {
            MeshPiece part;
            part.vmap.resize( vertsInPart );
            for ( const auto & t : tris )
            {
                auto v0p = t.v[0] / vertsInPart;
                auto v1p = t.v[1] / vertsInPart;
                auto v2p = t.v[2] / vertsInPart;

                if ( v0p != myPartId || v1p != myPartId || v2p != myPartId )
                {
                    // not my triangle

                    // thread 0 has to process border triangles
                    if ( myPartId == 0 && ( v0p != v1p || v1p != v2p || v2p != v0p ) )
                        borderTris.push_back( t );

                    continue;
                }

                VertId v[3] = {
                    VertId( t.v[0] % vertsInPart ),
                    VertId( t.v[1] % vertsInPart ),
                    VertId( t.v[2] % vertsInPart )
                };
                FaceId f{ part.tris.size() };
                part.tris.emplace_back( v[0], v[1], v[2], f );
                part.fmap.push_back( t.f );
                part.vmap[ v[0] ] = t.v[0];
                part.vmap[ v[1] ] = t.v[1];
                part.vmap[ v[2] ] = t.v[2];
            }
            part.topology = fromTrianglesSeq( part.tris );
            parts[myPartId] = std::move( part );
        }
    } );

    res = fromDisjointMeshPieces( parts, maxVertId, maxFaceId, borderTris );
    if ( skippedTris )
    {
        *skippedTris = std::move( borderTris );
    }
    return res;
}

static EdgeId smallestNoLeft( const MeshTopology & topology, VertId v )
{
    EdgeId res;
    if ( !topology.hasVert( v ) )
        return res;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        auto lf = topology.left( e );
        if ( lf )
            continue;
        if ( res && topology.left( res ) < lf )
            continue;
        res = e;
    }
    return res;
}

static void leaveOneStar( MeshTopology & topology, VertId v, std::vector<Triangle> & deleted )
{
    EdgeId nle = smallestNoLeft( topology, v );
    if ( !nle )
        return;
    assert( !topology.left( nle ) );
    EdgeId nre = nle;
    while ( topology.right( nre ) )
        nre = topology.prev( nre );
    for ( EdgeId e = topology.next( nle ); e != nre; e = topology.next( nle ) )
    {
        Triangle t;
        t.f = topology.left( e );
        assert( t.f );
        topology.getLeftTriVerts( e, t.v );
        deleted.push_back( t );
        deleteFace( topology, t.f );
    }
}

static void temporaryDeleteNonManifoldStars( MeshTopology & topology, std::vector<Triangle> & remainingTris )
{
    std::vector<Triangle> deleted;
    for ( auto & t : remainingTris )
    {
        for ( auto & v : t.v )
        {
            leaveOneStar( topology, v, deleted );
        }
    }
    if ( deleted.empty() )
        return;
    addTrianglesSeqCore( topology, remainingTris );
    remainingTris.insert( remainingTris.end(), deleted.begin(), deleted.end() );
    addTrianglesSeqCore( topology, remainingTris );
}

// two incident vertices can be found using this struct
struct IncidentVert {
    FaceId f; // to find triangle in triangleToVertices vector
    int cIdx = 0; // index of the central vertex in Triangle.v vector
    VertId srcVert; // used only for sort to speed up all search 
    // the vertices of the triangle can be upgraded, so no reason to store VertId!

    IncidentVert( FaceId f, int cIdx, VertId srcVert )
        : f(f)
        , cIdx(cIdx)
        , srcVert( srcVert )
    {}
};

using FaceToVerticesVector = Vector<std::array<VertId, 3>, FaceId>;

// to find the smallest connected sequences around central vertex, where a sequence does not repeat any neighbor vertex twice.
struct PathOverIncidentVert {
    FaceToVerticesVector& faceToVertices;
    // all iterators in [vertexBegIt, vertexEndIt) must have the same central vertex
    std::vector<IncidentVert>::iterator vertexBegIt, vertexEndIt;
    size_t lastUnvisitedIndex = 0; // pivot index. [vertexBegIt, vertexBegIt + lastUnvisitedIndex) - unvisited vertices

    PathOverIncidentVert( FaceToVerticesVector& triangleToVertices,
                std::vector<IncidentVert>& incidentItemsVector, size_t beg, size_t end )
        : faceToVertices( triangleToVertices )
        , vertexBegIt( incidentItemsVector.begin() + beg )
        , vertexEndIt( incidentItemsVector.begin() + end )
        , lastUnvisitedIndex( end - beg )
    {}

    // false if there are some unvisited vertices
    bool empty() const
    {
        return lastUnvisitedIndex <= 0;
    }

    // first unvisited vertex
    VertId getFirstVertex() const
    {
        return faceToVertices[vertexBegIt->f][( vertexBegIt->cIdx + 1 ) % 3];
    }

    // find incident unvisited vertex 
    VertId getNextIncidentVertex( VertId v )
    {
        if ( lastUnvisitedIndex <= 0 )
            return VertId( -1 );

        for ( auto it = vertexBegIt; it < vertexBegIt + lastUnvisitedIndex; ++it )
        {
            const auto& vertices = faceToVertices[it->f];
            VertId nextVertex( -1 );
            bool isIncident = false;
            for ( size_t i = 0; i < 3; ++i )
            {
                if ( i != it->cIdx )
                {
                    if ( vertices[i] != v )
                        nextVertex = vertices[i];
                    else
                        isIncident = true;
                }
            }
            if ( isIncident && nextVertex.valid() )
            {
                --lastUnvisitedIndex;
                std::iter_swap( it, vertexBegIt + lastUnvisitedIndex );
                return nextVertex;
            }

        }
        return VertId( -1 );
    }

    // duplicate the vertex around which the chain was found
    void duplicateVertex( std::vector<VertId>& path, int& newVertIndex,
                          std::vector<VertDuplication>* dups = nullptr )
    {
        if ( path.front() != path.back() )
            return;

        VertDuplication vertDup;
        vertDup.dupVert = VertId( ++newVertIndex );
        vertDup.srcVert = faceToVertices[vertexBegIt->f][vertexBegIt->cIdx];
        if ( dups )
            dups->push_back( vertDup );

        for ( size_t i = 1; i < path.size(); ++i )
        {
            for ( auto it = vertexBegIt; it < vertexEndIt; ++it )
            {
                size_t centalNum = it->cIdx;
                size_t firstNum = ( centalNum + 1 ) % 3;
                size_t secondNum = ( centalNum + 2 ) % 3;

                auto& vertices = faceToVertices[it->f];

                if ( ( vertices[firstNum] == path[i - 1] || vertices[secondNum] == path[i - 1] ) &&
                     ( vertices[firstNum] == path[i] || vertices[secondNum] == path[i] ) )
                {
                    vertices[it->cIdx] = vertDup.dupVert;
                }
            }
        }
    }
};

// from Triangle vector: fill FaceToVerticesVector ( to find all vertices by FaceId),
// fill and sort incidentVertVector by central vertex
void preprocessTriangles( std::vector<Triangle>& tris, std::vector<IncidentVert>& incidentVertVector,
                                     FaceToVerticesVector& faceToVertices )
{
    incidentVertVector.reserve( 3 * tris.size() );

    size_t maxFaceId = tris.size();
    for ( auto& tr : tris )
        maxFaceId = std::max( maxFaceId, static_cast< size_t >(tr.f.get()) + 1 );

    faceToVertices.resize( maxFaceId );
    for ( const auto& tr : tris )
    {
        std::copy( std::begin( tr.v ), std::end( tr.v ), std::begin( faceToVertices[tr.f] ) );
        for ( int i = 0; i < 3; ++i )
            incidentVertVector.emplace_back( tr.f, i, tr.v[i]);
    }

    std::sort( incidentVertVector.begin(), incidentVertVector.end(),
        [] ( const IncidentVert& lhv, const IncidentVert& rhv ) -> bool
    {
        return lhv.srcVert < rhv.srcVert;
    } );
}

// path = {abcDefgD} => closedPath = {DefgD}; path = {abc}
void extractClosedPath( std::vector<VertId>& path, std::vector<VertId>& closedPath )
{
    auto lastVertex = path.back();
    for ( size_t i = 0; i < path.size(); ++i )
    {
        if ( path[i] == lastVertex )
        {
            closedPath.reserve( path.size() - i );
            closedPath.insert( closedPath.end(), std::make_move_iterator( path.begin() + i ),
                                                 std::make_move_iterator( path.end() ) );

            path.resize(i);
            break;
        }
    }
}

// all IncidentVert in [posBegin, posEnd) have the same central vertex
size_t getIntervalEndIndex( const std::vector<IncidentVert>& incidentItemsVector,
                            const FaceToVerticesVector& faceToVertices,
                            size_t posBegin)
{
    size_t posEnd = posBegin + 1;
    auto prev = incidentItemsVector[posBegin];
    if ( posEnd < incidentItemsVector.size() )
    {
        auto curr = incidentItemsVector[posEnd];
        while ( posEnd < incidentItemsVector.size() &&
            faceToVertices[prev.f][prev.cIdx] == faceToVertices[curr.f][curr.cIdx] )
        {
            ++posEnd;
            if ( posEnd < incidentItemsVector.size() )
                curr = incidentItemsVector[posEnd];
        }
    }
    return posEnd;
}

// for all vertices get over all incident vertices to find connected sequences
size_t duplicateNonManifoldVertices( std::vector<Triangle>& tris, std::vector<VertDuplication>* dups )
{
    MR_TIMER
    FaceToVerticesVector faceToVertices;
    std::vector<IncidentVert> incidentItemsVector;
    preprocessTriangles( tris, incidentItemsVector, faceToVertices );

    auto lastIt = incidentItemsVector.rbegin();
    int maxIndex = faceToVertices[lastIt->f][lastIt->cIdx];

    size_t duplicatedVerticesCnt = 0;
    size_t posBegin = 0, posEnd = 0;
    while ( posEnd != incidentItemsVector.size() )
    {
        posBegin = posEnd;
        posEnd = getIntervalEndIndex( incidentItemsVector, faceToVertices, posBegin );
        PathOverIncidentVert incidentItems( faceToVertices, incidentItemsVector, posBegin, posEnd );

        size_t foundedPathCnt = 0;

        while ( !incidentItems.empty() )
        {
            VertId currVertex = incidentItems.getFirstVertex();
            VertId nextVertex = incidentItems.getNextIncidentVertex( currVertex );

            std::vector<VertId> path = { currVertex, nextVertex };
            while ( true )
            {
                currVertex = nextVertex;
                nextVertex = incidentItems.getNextIncidentVertex( currVertex );

                // no next vertex - not connected sequences
                if ( !nextVertex.valid() ) 
                    break;

                // returned to already visited vertex
                if ( std::find(path.begin(), path.end(), nextVertex ) != path.end() )
                {
                    // save only closed path and prepare for new search starting with non-manifold vertex
                    path.push_back( nextVertex );
                    std::vector<VertId> closedPath;
                    extractClosedPath( path, closedPath );

                    // if not first connected sequence found
                    if ( foundedPathCnt > 0 )
                    {
                        incidentItems.duplicateVertex( closedPath, maxIndex, dups );
                        ++duplicatedVerticesCnt;
                    }
                    ++foundedPathCnt;
                    
                }
                path.push_back( nextVertex );
            }
        }
    }
    // save modified vertices
    for ( auto& tr : tris )
    {
        const auto& vert = faceToVertices[tr.f];
        std::copy( std::begin( vert ), std::end( vert ), std::begin( tr.v ) );
    }
    return duplicatedVerticesCnt;
}

MeshTopology fromTrianglesDuplicatingNonManifoldVertices( const std::vector<Triangle> & tris,
    std::vector<VertDuplication> * dups, std::vector<Triangle> * skippedTris )
{
    MR_TIMER
    std::vector<Triangle> remainingTris;
    MeshTopology res = fromTriangles( tris, &remainingTris );
    HashMap<VertId, VertId> vert2duplicate;
    constexpr int MaxIter = 7;
    for ( int i = 0; i < MaxIter && !remainingTris.empty(); ++i )
    {
        vert2duplicate.clear();
        for ( auto & t : remainingTris )
        {
            for ( auto & v : t.v )
            {
                auto [it, inserted] = vert2duplicate.insert( std::make_pair( v, VertId{} ) );
                if ( inserted )
                {
                    // this vertex was not considered yet
                    if ( res.hasVert( v ) && !res.isBdVertex( v ) )
                    {
                        // full star found, make duplicate vertex
                        auto vd = res.addVertId();
                        if ( dups )
                            dups->push_back( { v, vd } );
                        it->second = v = vd;
                    }
                }
                else
                {
                    if ( it->second ) // if valid duplicate
                        v = it->second;
                }
            }
        }
        const auto sz0 = remainingTris.size();
        addTrianglesSeqCore( res, remainingTris );
        if ( sz0 == remainingTris.size() )
        {
            temporaryDeleteNonManifoldStars( res, remainingTris );
        }
    }
    if ( skippedTris )
        *skippedTris = std::move( remainingTris );
    return res;
}

MeshTopology fromVertexTriples( const std::vector<VertId> & vertTriples )
{
    MR_TIMER
    const size_t numTri = vertTriples.size() / 3;
    std::vector<Triangle> tris;
    tris.reserve( numTri );

    for ( size_t t = 0; t < numTri; ++t )
    {
        Triangle tri
        {
            vertTriples[3*t],
            vertTriples[3*t+1], 
            vertTriples[3*t+2],
            FaceId( int( t ) )
        };
        tris.push_back( tri );
    }
    return fromTriangles( tris );
}

// check non-manifold vertices resolving
TEST( MRMesh, duplicateNonManifoldVertices )
{
    std::vector<Triangle> tris;
    tris.emplace_back( 0_v, 1_v, 2_v, 0_f );
    tris.emplace_back( 0_v, 2_v, 3_v, 1_f );
    tris.emplace_back( 0_v, 3_v, 1_v, 2_f );

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( tris, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 0 );
    ASSERT_EQ( dups.size(), 0 );

    tris.emplace_back( 0_v, 4_v, 5_v, 3_f );
    tris.emplace_back( 0_v, 5_v, 6_v, 4_f );
    tris.emplace_back( 0_v, 6_v, 4_v, 5_f );

    duplicatedVerticesCnt = duplicateNonManifoldVertices( tris, &dups );
    ASSERT_EQ( duplicatedVerticesCnt,  1);
    ASSERT_EQ( dups.size(), 1 );
    ASSERT_EQ( dups[0].srcVert, 0 );
    ASSERT_EQ( dups[0].dupVert, 7 );

    int firstChangedTriangleNum = tris[0].v[0] != 0 ? 0 : 3;
    for ( int i = firstChangedTriangleNum; i < firstChangedTriangleNum + 3; ++i )
        ASSERT_EQ( tris[i].v[0], 7 );
}

} //namespace MeshBuilder

} //namespace MR
