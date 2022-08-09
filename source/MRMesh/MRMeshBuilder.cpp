#include "MRMeshBuilder.h"
#include "MRIdentifyVertices.h"
#include "MRRingIterator.h"
#include "MRAABBTreePoints.h"
#include "MRPointsInBall.h"
#include "MRTimer.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace MeshBuilder
{

static VertId findMaxVertId( const Triangulation & t, const FaceBitSet * region )
{
    MR_TIMER
    return parallel_reduce( tbb::blocked_range( 0_f, t.endId() ), VertId{},
    [&] ( const auto & range, VertId currMax )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
        {
            if ( region && !region->test( f ) )
                continue;
            currMax = std::max( { currMax, t[f][0], t[f][1], t[f][2] } );
        }
        return currMax;
    },
    [] ( VertId a, VertId b )
    {
        return a > b ? a : b;
    } );
}

static FaceId findMaxFaceId( const std::vector<Triangle> & tris )
{
    MR_TIMER
    return parallel_reduce( tbb::blocked_range( tris.begin(), tris.end() ), FaceId{},
    [&] ( const auto & range, FaceId currMax )
    {
        for ( const Triangle & t : range )
        {
            currMax = std::max( currMax, t.f );
        }
        return currMax;
    },
    [] ( FaceId a, FaceId b )
    {
        return a > b ? a : b;
    } );
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

static FaceBitSet getLocalRegion( FaceBitSet * region, size_t tSize )
{
    FaceBitSet res;
    if ( region )
        res = std::move( *region );
    else
    {
        res = FaceBitSet( tSize );
        res.set();
    }
    return res;
}

static void addTrianglesSeqCore( MeshTopology& res, const Triangulation & t, const BuildSettings & settings = {} )
{
    MR_TIMER

    FaceAdder fa;
    // we will try to add these triangles in the current pass
    FaceBitSet active = getLocalRegion( settings.region, t.size() );
    // these are triangles that cannot be added even after other triangles
    FaceBitSet bad;
    for (;;)
    {
        size_t triAddedOnThisPass = 0;
        for ( FaceId f : active )
        {
            auto x = fa.add( res, f + settings.shiftFaceId, t[f].data(), t[f].data() + 3, settings.allowNonManifoldEdge );
            if ( x == AddFaceResult::UnsafeTryLater )
                continue;
            active.reset( f );
            if ( x != AddFaceResult::Success )
                bad.autoResizeSet( f );
            else
                ++triAddedOnThisPass;
        }

        if ( triAddedOnThisPass == 0 )
            break; // no single triangle added during the pass
    }
    if ( settings.region )
    {
        active |= bad;
        *settings.region = std::move( active );
    }
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

void addTriangles( MeshTopology & res, const Triangulation & t, const BuildSettings & settings )
{
    MR_TIMER
    if ( t.empty() )
        return;

    // reserve enough elements for faces and vertices
    const auto maxVertId = findMaxVertId( t, settings.region );
    res.faceResize( t.size() );
    res.vertResize( maxVertId + 1 );

    addTrianglesSeqCore( res, t, settings );
}

void addTriangles( MeshTopology & res, std::vector<VertId> & vertTriples,
    FaceBitSet * createdFaces )
{
    MR_TIMER

    const int numTri = (int)vertTriples.size() / 3;
    Triangulation t;
    t.reserve( numTri );
    const auto firstNewFace = res.lastValidFace() + 1;

    for ( int f = 0; f < numTri; ++f )
        t.push_back( { vertTriples[3*f], vertTriples[3*f+1], vertTriples[3*f+2] } );

    if ( createdFaces )
    {
        if ( createdFaces->size() <= firstNewFace + numTri )
            createdFaces->resize( firstNewFace + numTri + 1 );
        createdFaces->set( firstNewFace, numTri, true );
    }

    vertTriples.clear();
    FaceBitSet region( numTri, 1 );
    addTriangles( res, t, { .region = &region, .shiftFaceId = firstNewFace } );
    for ( FaceId f : region )
    {
        vertTriples.push_back( t[f][0] );
        vertTriples.push_back( t[f][1] );
        vertTriples.push_back( t[f][2] );
        if ( createdFaces )
            createdFaces->reset( f );
    }
}

static MeshTopology fromTrianglesSeq( const Triangulation & t, const BuildSettings & settings )
{
    MeshTopology res;
    addTriangles( res, t, settings );
    return res;
}

MeshTopology fromDisjointMeshPieces( const Triangulation & t, VertId maxVertId,
    const std::vector<MeshPiece> & pieces,
    const BuildSettings & settings0 )
{
    MR_TIMER

    // construct empty final mesh with enough number of elements
    const auto numParts = pieces.size();
    std::vector<EdgeId> firstPartEdge( numParts + 1 );
    firstPartEdge[0] = 0_e;
    FaceBitSet region;
    if ( settings0.region )
        region = std::move( *settings0.region );
    region.resize( t.size() );
    for ( int i = 0; i < pieces.size(); ++i )
    {
        const auto & p = pieces[i];
        firstPartEdge[i + 1] = firstPartEdge[i] + (int)p.topology.edgeSize();
        for ( FaceId f : p.rem ) // remaining triangles
            region.set( p.fmap[f] );
    }
    int numEdgesInParts = firstPartEdge.back();

    MeshTopology res;
    const auto borderTriCount = settings0.region ? settings0.region->count() : 0;
    res.edgeReserve( numEdgesInParts + 6 * borderTriCount ); // should be enough even if all border triangles are disconnected
    res.resizeBeforeParallelAdd( numEdgesInParts, maxVertId + 1, t.size() );

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
    BuildSettings settings = settings0;
    settings.region = &region;
    addTrianglesSeqCore( res, t, settings );
    if ( settings0.region )
        *settings0.region = std::move( region );

    return res;
}

MeshTopology fromTriangles( const std::vector<Triangle> & tris, std::vector<Triangle> * skippedTris )
{
    if ( tris.empty() )
        return {};
    MR_TIMER

    const auto maxFaceId = findMaxFaceId( tris );
    Triangulation triangulation( maxFaceId + 1 );
    FaceBitSet region( maxFaceId + 1 );
    for ( const auto & t : tris )
    {
        triangulation[t.f] = t.v;
        region.set( t.f );
    }

    auto res = fromTriangles( triangulation, { .region = &region } );
    if ( skippedTris )
    {
        skippedTris->clear();
        for ( auto f : region )
            skippedTris->push_back( tris[f] );
    }
    return res;
}

MeshTopology fromTriangles( const Triangulation & t, const BuildSettings & settings )
{
    if ( t.empty() )
        return {};
    MR_TIMER
    
    // reserve enough elements for faces and vertices
    //auto [maxFaceId, maxVertId] = computeMaxIds( tris );
    const auto maxVertId = findMaxVertId( t, settings.region );

    constexpr size_t minTrisInPart = 32768;
    // numParts shall not depend on hardware (e.g. on std::thread::hardware_concurrency()) to be repeatable on all hardware
    const size_t numParts = std::min( ( t.size() + minTrisInPart - 1 ) / minTrisInPart, (size_t)64 );
    assert( numParts >= 1 );
    MeshTopology res;
    if ( numParts <= 1 )
        return fromTrianglesSeq( t, settings );

    const size_t vertsInPart = ( (int)maxVertId + numParts ) / numParts;
    std::vector<MeshPiece> parts( numParts );
    FaceBitSet borderTris( t.size() ); // triangles having vertices in distinct parts, computed by thread 0

    Timer timer("parallel parts");
    // construct mesh parts
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, numParts, 1 ), [&]( const tbb::blocked_range<size_t> & range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t myPartId = range.begin(); myPartId < range.end(); ++myPartId )
        {
            MeshPiece part;
            Triangulation partTriangulation;
            BuildSettings partSettings{ .region = &part.rem, .allowNonManifoldEdge = settings.allowNonManifoldEdge };
            part.vmap.resize( vertsInPart );
            for ( FaceId f{0}; f < t.size(); ++f )
            {
                if ( settings.region && !settings.region->test( f ) )
                    continue;
                const auto & vs = t[f];
                auto v0p = vs[0] / vertsInPart;
                auto v1p = vs[1] / vertsInPart;
                auto v2p = vs[2] / vertsInPart;

                if ( v0p != myPartId || v1p != myPartId || v2p != myPartId )
                {
                    // not my triangle

                    // thread 0 has to process border triangles
                    if ( myPartId == 0 && ( v0p != v1p || v1p != v2p || v2p != v0p ) )
                        borderTris.set( f );

                    continue;
                }

                VertId v[3] = {
                    VertId( vs[0] % vertsInPart ),
                    VertId( vs[1] % vertsInPart ),
                    VertId( vs[2] % vertsInPart )
                };
                FaceId fp{ partTriangulation.size() };
                partTriangulation.push_back( ThreeVertIds{ v[0], v[1], v[2] } );
                part.fmap.push_back( f );
                part.vmap[ v[0] ] = vs[0];
                part.vmap[ v[1] ] = vs[1];
                part.vmap[ v[2] ] = vs[2];
                part.rem.autoResizeSet( fp );
            }
            part.topology = fromTrianglesSeq( partTriangulation, partSettings );
            parts[myPartId] = std::move( part );
        }
    } );

    auto joinSettings = settings;
    joinSettings.region = &borderTris;
    res = fromDisjointMeshPieces( t, maxVertId, parts, joinSettings );
    if ( settings.region )
        *settings.region = std::move( borderTris );
    return res;
}

// two incident vertices can be found using this struct
struct IncidentVert {
    FaceId f; // to find triangle in triangleToVertices vector
    VertId srcVert; // central vertex, used for sorting triangles per their incident vertices
    // the vertices of the triangle can be upgraded, so no reason to store VertId!

    IncidentVert( FaceId f, VertId srcVert )
        : f(f)
        , srcVert( srcVert )
    {}
};

// to find the smallest connected sequences around central vertex, where a sequence does not repeat any neighbor vertex twice.
struct PathOverIncidentVert {
    Triangulation& faceToVertices;
    // all iterators in [vertexBegIt, vertexEndIt) must have the same central vertex
    std::vector<IncidentVert>::iterator vertexBegIt, vertexEndIt;
    size_t lastUnvisitedIndex = 0; // pivot index. [vertexBegIt, vertexBegIt + lastUnvisitedIndex) - unvisited vertices

    PathOverIncidentVert( Triangulation& triangleToVertices,
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
        for ( auto v : faceToVertices[vertexBegIt->f] )
            if ( v != vertexBegIt->srcVert )
                return v;
        assert( false );
        return {};
    }

    // find incident unvisited vertex
    VertId getNextIncidentVertex( VertId v, bool triOrientation )
    {
        if ( lastUnvisitedIndex <= 0 )
            return VertId( -1 );

        for ( auto it = vertexBegIt; it < vertexBegIt + lastUnvisitedIndex; ++it )
        {
            VertId nextVertex;
            const auto & vs = faceToVertices[it->f];
            if ( triOrientation )
            {
                if ( vs[0] == it->srcVert && vs[1] == v )
                    nextVertex = vs[2];
                else if ( vs[1] == it->srcVert && vs[2] == v )
                    nextVertex = vs[0];
                else if ( vs[2] == it->srcVert && vs[0] == v )
                    nextVertex = vs[1];
            }
            else
            {
                if ( vs[1] == it->srcVert && vs[0] == v )
                    nextVertex = vs[2];
                else if ( vs[2] == it->srcVert && vs[1] == v )
                    nextVertex = vs[0];
                else if ( vs[0] == it->srcVert && vs[2] == v )
                    nextVertex = vs[1];
            }
            if ( nextVertex )
            {
                --lastUnvisitedIndex;
                std::iter_swap( it, vertexBegIt + lastUnvisitedIndex );
                return nextVertex;
            }

        }
        return {};
    }

    // duplicate the vertex around which the chain was found
    void duplicateVertex( std::vector<VertId>& path, VertId& lastUsedVertId,
                          std::vector<VertDuplication>* dups = nullptr )
    {
        VertDuplication vertDup;
        vertDup.dupVert = ++lastUsedVertId;
        vertDup.srcVert = vertexBegIt->srcVert;
        if ( dups )
            dups->push_back( vertDup );

        for ( size_t i = 1; i < path.size(); ++i )
        {
            for ( auto it = vertexBegIt + lastUnvisitedIndex; it < vertexEndIt; ++it )
            {
                VertId v1, v2;
                bool alreadyDuplicted = true;
                for ( VertId vi : faceToVertices[it->f] )
                {
                    if ( vi == vertDup.srcVert )
                        alreadyDuplicted = false;
                    else if ( !v1 )
                        v1 = vi;
                    else if ( !v2 )
                        v2 = vi;
                }
                if ( alreadyDuplicted )
                    continue;
                assert( v1 && v2 );

                if ( ( v1 == path[i - 1] || v2 == path[i - 1] ) &&
                     ( v1 == path[i] || v2 == path[i] ) )
                {
                    for ( VertId & vi : faceToVertices[it->f] )
                    {
                        if ( vi != vertDup.srcVert )
                            continue;
                        vi = vertDup.dupVert;
                        break;
                    }
                    it->srcVert = vertDup.dupVert;
                    break;
                }
            }
        }
    }
};

// fill and sort incidentVertVector by central vertex
void preprocessTriangles( const Triangulation & t, FaceBitSet * region, std::vector<IncidentVert>& incidentVertVector )
{
    incidentVertVector.reserve( 3 * t.size() );

    for ( FaceId f{0}; f < t.size(); ++f )
    {
        if ( region && !region->test( f ) )
            continue;
        const auto & vs = t[f];
        if ( vs[0] == vs[1] || vs[1] == vs[2] || vs[2] == vs[0] )
            continue;

        for ( int i = 0; i < 3; ++i )
            incidentVertVector.emplace_back( f, vs[i] );
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
    closedPath.clear();
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

// for all vertices get over all incident vertices to find connected sequences
size_t duplicateNonManifoldVertices( Triangulation & t, FaceBitSet * region, std::vector<VertDuplication>* dups )
{
    MR_TIMER
    if ( t.empty() )
        return 0;

    std::vector<IncidentVert> incidentItemsVector;
    preprocessTriangles( t, region, incidentItemsVector );

    auto lastUsedVertId = incidentItemsVector.back().srcVert;

    std::vector<VertId> path;
    std::vector<VertId> closedPath;
    VertBitSet visitedVertices(lastUsedVertId);
    size_t duplicatedVerticesCnt = 0;
    size_t posBegin = 0, posEnd = 0;
    while ( posEnd != incidentItemsVector.size() )
    {
        posBegin = posEnd++;
        while ( posEnd < incidentItemsVector.size() && incidentItemsVector[posBegin].srcVert == incidentItemsVector[posEnd].srcVert )
            ++posEnd;
        PathOverIncidentVert incidentItems( t, incidentItemsVector, posBegin, posEnd );

        // first chain of vertices around the center does not require duplication
        int foundChains = 0;
        while ( !incidentItems.empty() )
        {
            for(const auto& v : path)
                visitedVertices.reset(v);

            bool triOrientation = true;
            const VertId firstVertex = incidentItems.getFirstVertex();
            visitedVertices.autoResizeSet( firstVertex );
            VertId nextVertex = incidentItems.getNextIncidentVertex( firstVertex, triOrientation );
            if ( !nextVertex )
            {
                triOrientation = false;
                nextVertex = incidentItems.getNextIncidentVertex( firstVertex, triOrientation );
                assert( nextVertex.valid() );
            }
            visitedVertices.autoResizeSet( nextVertex );

            path = { firstVertex, nextVertex };
            while ( true )
            {
                nextVertex = incidentItems.getNextIncidentVertex( nextVertex, triOrientation );

                if ( !nextVertex )
                {
                    if ( triOrientation ) // try the opposite direction from firstVertex
                    {
                        triOrientation = false;
                        nextVertex = incidentItems.getNextIncidentVertex( firstVertex, triOrientation );
                    }
                    if ( !nextVertex )
                    {
                        if ( foundChains )
                        {
                            incidentItems.duplicateVertex( path, lastUsedVertId, dups );
                            ++duplicatedVerticesCnt;
                        }
                        ++foundChains;
                        break;
                    }
                    std::reverse( path.begin(), path.end() );
                }

                // returned to already visited vertex
                if ( visitedVertices.test(nextVertex) )
                {
                    // save only closed path and prepare for new search starting with non-manifold vertex
                    path.push_back( nextVertex );
                    extractClosedPath( path, closedPath );
                    for( const auto& v : closedPath)
                        visitedVertices.reset(v);

                    if ( foundChains )
                    {
                        incidentItems.duplicateVertex( closedPath, lastUsedVertId, dups );
                        ++duplicatedVerticesCnt;
                    }
                    ++foundChains;
                    if ( path.empty() )
                        break;
                }
                path.push_back( nextVertex );
                visitedVertices.autoResizeSet( nextVertex );
            }
        }
    }
    return duplicatedVerticesCnt;
}

MeshTopology fromTrianglesDuplicatingNonManifoldVertices( Triangulation & t,
    std::vector<VertDuplication> * dups, const BuildSettings & settings )
{
    MR_TIMER
    FaceBitSet localRegion = getLocalRegion( settings.region, t.size() );
    BuildSettings localSettings = settings;
    localSettings.region = &localRegion;
    // try happy path first
    MeshTopology res = fromTriangles( t, localSettings );
    if ( !localRegion.any() )
    {
        // all triangles added successfully, which means no non-manifold vertices
        if ( dups )
            dups->clear();
        if ( settings.region )
            settings.region->clear();
        return res;
    }
    // full path
    std::vector<VertDuplication> localDups;
    MeshBuilder::duplicateNonManifoldVertices( t, settings.region, &localDups );
    const bool noDuplicates = localDups.empty();
    if ( dups )
        *dups = std::move( localDups );
    if ( noDuplicates )
    {
        // no duplicates created, so res is ok
        if ( settings.region )
            settings.region->clear();
        return res;
    }
    res = fromTriangles( t, settings );
    return res;
}

MeshTopology fromVertexTriples( const std::vector<VertId> & vertTriples )
{
    MR_TIMER
    const size_t numTri = vertTriples.size() / 3;
    Triangulation t;
    t.reserve( numTri );
    
    for ( size_t f = 0; f < numTri; ++f )
    {
        t.push_back( { vertTriples[3*f], vertTriples[3*f+1], vertTriples[3*f+2] } );
    }
    return fromTriangles( t );
}

Mesh fromPointTriples( const std::vector<ThreePoints> & posTriples )
{
    MR_TIMER
    VertexIdentifier vi;
    vi.reserve( posTriples.size() );
    vi.addTriangles( posTriples );
    Mesh res;
    res.points = vi.takePoints();
    res.topology = fromTriangles( vi.takeTriangulation() );
    return res;
}

int uniteCloseVertices( Mesh & mesh, float closeDist, bool uniteOnlyBd, VertMap * optionalVertOldToNew )
{
    MR_TIMER
    VertBitSet bdVerts;
    if ( uniteOnlyBd )
        bdVerts = mesh.topology.findBoundaryVerts();

    AABBTreePoints tree( mesh.points, uniteOnlyBd ? bdVerts : mesh.topology.getValidVerts() );
    VertMap vertOldToNew( mesh.topology.vertSize() );
    int numChanged = 0;
    for ( VertId v : mesh.topology.getValidVerts() )
    {
        VertId smallestCloseVert = v;
        if ( !uniteOnlyBd || bdVerts.test( v ) )
        {
            findPointsInBall( tree, mesh.points[v], closeDist, [&]( VertId cv, const Vector3f& )
            {
                if ( cv == v )
                    return;
                if ( vertOldToNew[cv] != cv )
                    return; // cv vertex is removed by itself
                smallestCloseVert = std::min( smallestCloseVert, cv );
            } );
        }
        vertOldToNew[v] = smallestCloseVert;
        if ( v != smallestCloseVert )
            ++numChanged;
    }
    if ( numChanged <= 0 )
        return numChanged;

    Triangulation t( mesh.topology.faceSize() );
    FaceBitSet region( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        ThreeVertIds oldt, newt;
        mesh.topology.getTriVerts( f, oldt );
        for ( int i = 0; i < 3; ++i )
            newt[i] = vertOldToNew[oldt[i]];
        if ( oldt != newt )
        {
            mesh.topology.deleteFace( f );
            t[f] = newt;
            region.set( f );
        }
    }
    addTriangles( mesh.topology, t, { .region = &region } );
    mesh.invalidateCaches();
    if ( optionalVertOldToNew )
        *optionalVertOldToNew = std::move( vertOldToNew );

    return numChanged;
}

// check non-manifold vertices resolving
TEST( MRMesh, duplicateNonManifoldVertices )
{
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } ); //0_f
    t.push_back( { 0_v, 2_v, 3_v } ); //1_f
    t.push_back( { 0_v, 3_v, 1_v } ); //2_f

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 0 );
    ASSERT_EQ( dups.size(), 0 );

    t.push_back( { 0_v, 4_v, 5_v } ); //3_f
    t.push_back( { 0_v, 5_v, 6_v } ); //4_f
    t.push_back( { 0_v, 6_v, 4_v } ); //5_f

    duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 1 );
    ASSERT_EQ( dups.size(), 1 );
    ASSERT_EQ( dups[0].srcVert, 0 );
    ASSERT_EQ( dups[0].dupVert, 7 );

    int firstChangedTriangleNum = t[0_f][0] != 0 ? 0 : 3;
    for ( FaceId i{ firstChangedTriangleNum }; i < firstChangedTriangleNum + 3; ++i )
        ASSERT_EQ( t[i][0], 7 );
}

} //namespace MeshBuilder

} //namespace MR
