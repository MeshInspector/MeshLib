#include "MRVertDuplication.h"
#include "MRBitSet.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRVector.h"
#include "MRphmap.h"
#include "MRPch/MRTBB.h"
#include <algorithm>

namespace MR
{

namespace MeshBuilder
{

/// returns the two other triangle vertices in cyclic order following vertex v
static std::pair<VertId, VertId> getOtherTriVerts( const ThreeVertIds & vs, VertId v )
{
    if ( vs[0] == v )
        return { vs[1], vs[2] };
    if ( vs[1] == v )
        return { vs[2], vs[0] };
    assert( vs[2] == v );
    return { vs[0], vs[1] };
}

// to find connected sequences around central vertex, where a sequence does not repeat any neighbor vertex twice.
class PathAroundVertex
{
    Triangulation* faceToVertices = nullptr;
    const std::vector<VertTri>* vertTris = nullptr;
    // all elements in [vertexBegIndex, vertexEndIndex) of *vertTris have the same central vertex
    size_t vertexBegIndex = 0, vertexEndIndex = 0;
    size_t firstUnvisitedIndex = 0; // lazily advanced index of the first element with not yet visited face
    VertId center; // the central vertex of the current neighborhood

    // (vnext, f): there is triangle #f with the vertices (center, vnext, some-third-vert) up to rotation
    std::vector<std::pair<VertId, FaceId>> vnextFaces;
    // (vprev, f): there is triangle #f with the vertices (vprev, center, some-third-vert) up to rotation
    std::vector<std::pair<VertId, FaceId>> vprevFaces;
    HashSet<FaceId> visitedFaces;

public:
    // prepares the search around the central vertex of elements [beg, end), reusing the memory allocated for a previous vertex
    void init( Triangulation& triangleToVertices, const std::vector<VertTri>& tris, size_t beg, size_t end )
    {
        faceToVertices = &triangleToVertices;
        vertTris = &tris;
        vertexBegIndex = beg;
        vertexEndIndex = end;
        firstUnvisitedIndex = beg;
        assert( beg < end );
        center = tris[beg].v;

        vnextFaces.clear();
        vprevFaces.clear();
        visitedFaces.clear();
        vnextFaces.reserve( end - beg );
        vprevFaces.reserve( end - beg );
        for ( auto i = beg; i < end; ++i )
        {
            assert( tris[i].v == center );
            const auto f = tris[i].f;
            const auto [v1, v2] = getOtherTriVerts( triangleToVertices[f], center );
            vnextFaces.emplace_back( v1, f );
            vprevFaces.emplace_back( v2, f );
        }
        std::sort( vnextFaces.begin(), vnextFaces.end() );
        std::sort( vprevFaces.begin(), vprevFaces.end() );
    }

    // false if there are some elements with not yet visited faces; advances firstUnvisitedIndex
    bool empty()
    {
        while ( firstUnvisitedIndex < vertexEndIndex && visitedFaces.contains( (*vertTris)[firstUnvisitedIndex].f ) )
            ++firstUnvisitedIndex;
        return firstUnvisitedIndex >= vertexEndIndex;
    }

    // takes the first triangle with not yet visited face and returns its two other vertices in cyclic order
    // to start a new path there, so the walk can continue with triOrientation = true
    std::pair<VertId, VertId> getFirstTwoVertices()
    {
        [[maybe_unused]] const bool noRecs = empty(); // also advances firstUnvisitedIndex
        assert( !noRecs );
        const auto f = (*vertTris)[firstUnvisitedIndex].f;
        visitedFaces.insert( f );
        return getOtherTriVerts( (*faceToVertices)[f], center );
    }

    // find incident vertex in a not yet visited face except for prevVertex and its duplicates
    VertId getNextVertex( VertId v, bool triOrientation, VertId prevVertex, const std::vector<VertDuplication>& dups )
    {
        if ( empty() )
            return {};

        // if v is an original vertex, then return it;
        // if v is a duplicated vertex, then return the id of the original vertex, which was duplicated to make v
        auto getOrgVertex = [&dups]( VertId v )
        {
            if ( dups.empty() || v < dups.front().dupVert )
                return v;
            const auto i = v - dups.front().dupVert;
            assert( i < dups.size() );
            if ( i >= dups.size() )
                return v;
            assert( dups[i].dupVert == v );
            assert( dups[i].srcVert < dups.front().dupVert );
            return dups[i].srcVert;
        };

        assert( prevVertex );
        prevVertex = getOrgVertex( prevVertex );
        assert( prevVertex );

        const auto & vec = triOrientation ? vnextFaces : vprevFaces;
        for ( auto it = std::lower_bound( vec.begin(), vec.end(), std::make_pair( v, FaceId{} ) );
              it != vec.end() && it->first == v; ++it )
        {
            const auto f = it->second;
            if ( visitedFaces.contains( f ) )
                continue;
            const auto v12 = getOtherTriVerts( (*faceToVertices)[f], center );
            assert( ( triOrientation ? v12.first : v12.second ) == v );
            const auto nextVertex = triOrientation ? v12.second : v12.first;
            if ( getOrgVertex( nextVertex ) != prevVertex )
            {
                visitedFaces.insert( f );
                return nextVertex;
            }
        }
        return {};
    }

    // duplicate the vertex around which the chain was found
    void duplicateVertex( VertId v, const std::vector<VertId>& path, VertId& lastUsedVertId, bool triOrientation,
                          std::vector<VertDuplication>* dups = nullptr )
    {
        assert( v == center );
        VertDuplication vertDup;
        vertDup.dupVert = ++lastUsedVertId;
        vertDup.srcVert = v;
        if ( dups )
            dups->push_back( vertDup );

        [[maybe_unused]] size_t changedTris = 0;
        const auto & vec = triOrientation ? vnextFaces : vprevFaces;
        for ( size_t i = 1; i < path.size(); ++i )
        {
            // the triangle of this path step is (srcVert, path[i-1], path[i]) for triOrientation = true,
            // and (srcVert, path[i], path[i-1]) otherwise, up to rotation
            for ( auto it = std::lower_bound( vec.begin(), vec.end(), std::make_pair( path[i - 1], FaceId{} ) );
                  it != vec.end() && it->first == path[i - 1]; ++it )
            {
                const auto f = it->second;
                if ( !visitedFaces.contains( f ) )
                    continue; // only visited triangles can be in the path
                auto & tri = (*faceToVertices)[f];
                if ( tri[0] != vertDup.srcVert && tri[1] != vertDup.srcVert && tri[2] != vertDup.srcVert )
                    continue; // this triangle has already been re-pointed to the duplicate
                const auto v12 = getOtherTriVerts( tri, vertDup.srcVert );
                if ( ( triOrientation ? v12.second : v12.first ) != path[i] )
                    continue;
                for ( VertId & vi : tri )
                {
                    if ( vi != vertDup.srcVert )
                        continue;
                    vi = vertDup.dupVert;
                    break;
                }
                ++changedTris;
                break;
            }
        }
        assert( changedTris + 1 == path.size() );
    }
};

class VertNeighbourhoodInspector
{
public:
    VertInfo run( const Triangulation & t, const VertTri * begin, const VertTri * end );

private:
    struct VertRepetitions
    {
        VertId v;
        std::uint32_t r = 0;
    };
    static_assert( sizeof( VertRepetitions ) == 8 );

    /// l_[v1] is present in the map, if there is a triangle to the left of (v,v1) edge;
    /// l_[v1].v is invalid if there is a triangle to the right of (v,v1) edge;
    /// otherwise it is the vertex v2 such that there is a chain of triangles in between (v,v1) and (v,v2) and there is no triangle to the left of (v,v2) edge
    HashMap<VertId, VertRepetitions> l_;

    /// r_[v2] is present in the map, if there is a triangle to the right of (v,v2) edge;
    /// r_[v2].v is invalid if there is a triangle to the left of (v,v2) edge;
    /// otherwise it is the vertex v1 such that there is a chain of triangles in between (v,v1) and (v,v2) and there is no triangle to the right of (v,v1) edge
    HashMap<VertId, VertRepetitions> r_;
};

VertInfo inspectVertNeighbourhood( const Triangulation & t, const VertTri * begin, const VertTri * end )
{
    return VertNeighbourhoodInspector{}.run( t, begin, end );
}

VertInfo VertNeighbourhoodInspector::run( const Triangulation & t, const VertTri * begin, const VertTri * end )
{
    l_.clear();
    r_.clear();
    if ( begin == end )
        return {};
    const auto v0 = begin->v;
    std::uint32_t repeatedVerts = 0, maxVertRepetitions = 0;
    std::uint32_t openChains = 0, closedChains = 0;
    for ( auto i = begin; i != end; ++i )
    {
        assert( i->v == v0 );
        const auto [v1, v2] = getOtherTriVerts( t[i->f], v0 );
        const auto lInsertion = l_.insert( { v1, { v2 } } );
        const auto rInsertion = r_.insert( { v2, { v1 } } );
        if ( repeatedVerts == 0 && lInsertion.second && rInsertion.second )
        {
            ++openChains;
            if ( auto it = l_.find( v2 ); it != l_.end() )
            {
                // the edge (v,v2) becomes inner
                const auto vEnd = it->second.v;
                it->second.v = VertId{};
                assert( vEnd ); // the edge (v,v2) was boundary
                --openChains;
                lInsertion.first->second.v = vEnd;
                assert( r_[vEnd].v == v2 );
                r_[vEnd].v = v1;
            }
            if ( auto it = r_.find( v1 ); it != r_.end() )
            {
                // the edge (v,v1) becomes inner
                const auto vEnd = it->second.v;
                it->second.v = VertId{};
                assert( vEnd ); // the edge (v,v1) was boundary
                if ( vEnd == v1 )
                {
                    // the chain is closed
                    assert( lInsertion.first->second.v == v1 );
                    lInsertion.first->second.v = VertId{};
                    rInsertion.first->second.v = VertId{};
                    --openChains;
                    ++closedChains;
                }
                else
                {
                    --openChains;
                    // the right end of the chain grown from the current triangle: v2, or updated by the merge above
                    const auto vRight = lInsertion.first->second.v;
                    assert( vRight );
                    if ( vRight != v2 )
                    {
                        // the current triangle merged two chains on both sides, so its both edges are inner
                        lInsertion.first->second.v = VertId{};
                        rInsertion.first->second.v = VertId{};
                        assert( r_[vRight].v == v1 );
                        r_[vRight].v = vEnd;
                    }
                    else
                        rInsertion.first->second.v = vEnd;
                    assert( l_[vEnd].v == v1 );
                    l_[vEnd].v = vRight;
                }
            }
        }
        else
        {
            // insertion can fail only if the vertex is repeated
            if ( !lInsertion.second )
            {
                ++repeatedVerts;
                maxVertRepetitions = std::max( maxVertRepetitions, ++lInsertion.first->second.r );
            }
            if ( !rInsertion.second )
            {
                ++repeatedVerts;
                maxVertRepetitions = std::max( maxVertRepetitions, ++rInsertion.first->second.r );
            }
        }
    }
    VertInfo info;
    if ( repeatedVerts == 0 )
        info.setNumChains( openChains, closedChains );
    else
        info.setNumRepeatedVerts( repeatedVerts, maxVertRepetitions );
    return info;
}

struct AllVertTris
{
    /// the array of all vertex-in-triangle sorted by vertex id, then by face id
    std::vector<VertTri> recs;

    /// initializes recs
    AllVertTris( const Triangulation & t, const FaceBitSet * region );

    /// maps vertex id to first its record in recs, not descending;
    /// vertex #i is in the records [vert2firstRec[i], vert2firstRec[i+1]) of recs
    Vector<int, VertId> vert2firstRec;

    /// fills vert2firstRec
    void computeVertSpans();

    /// manifoldness info for each vertex
    Vector<VertInfo, VertId> vertInfos;

    /// fills vertInfos
    void computeVertInfos( const Triangulation & t );
};

AllVertTris::AllVertTris( const Triangulation & t, const FaceBitSet * region )
{
    MR_TIMER;

    if ( region )
        recs.reserve( 3 * region->count() );
    else
        recs.reserve( 3 * t.size() );

    for ( FaceId f{0}; f < t.size(); ++f )
    {
        if ( region && !region->test( f ) )
            continue;
        const auto & vs = t[f];
        if ( vs[0] == vs[1] || vs[1] == vs[2] || vs[2] == vs[0] )
            continue;

        for ( int i = 0; i < 3; ++i )
            recs.push_back( { vs[i], f } );
    }

    tbb::parallel_sort( recs.begin(), recs.end() );
}

void AllVertTris::computeVertSpans()
{
    MR_TIMER;
    if ( recs.empty() )
        return;

    vert2firstRec.reserve( recs.back().v + 2 );
    for ( int i = 0; i < recs.size(); ++i )
    {
        auto v = recs[i].v;
        while ( v >= vert2firstRec.size() )
            vert2firstRec.push_back( i );
    }
    vert2firstRec.push_back( (int)recs.size() );
    assert( vert2firstRec.size() == recs.back().v + 2 );
}

void AllVertTris::computeVertInfos( const Triangulation & t )
{
    MR_TIMER;
    if ( vert2firstRec.empty() )
        return;
    vertInfos.clear();
    vertInfos.resize( vert2firstRec.size() - 1 );

    tbb::enumerable_thread_specific<VertNeighbourhoodInspector> e;
    ParallelFor( vertInfos, e, [&]( VertId v, VertNeighbourhoodInspector & td )
    {
        vertInfos[v] = td.run( t, recs.data() + vert2firstRec[v], recs.data() + vert2firstRec[v + 1] );
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
size_t duplicateNonManifoldVertices( Triangulation & t, FaceBitSet * region, std::vector<VertDuplication>* dups, VertId lastValidVert )
{
    MR_TIMER;
    if ( dups )
        dups->clear(); // input contents are ignored
    if ( t.empty() )
        return 0; // input triangulation is empty

    AllVertTris all( t, region );
    if ( all.recs.empty() )
        return 0; // input triangulation contains only degenerate triangles, e.g. with repeating vertex (v v u)

    // maintain the duplications even if the caller did not ask for them, they are necessary for getOrgVertex;
    // the caller's vector was cleared above, moving it in just reuses its buffer
    std::vector<VertDuplication> myDups;
    if ( dups )
        myDups = std::move( *dups );

    if ( !lastValidVert )
        lastValidVert = all.recs.back().v;

    all.computeVertSpans();
    all.computeVertInfos( t );

    // collect the vertices requiring duplication in the original triangulation;
    // a vertex not requiring duplication cannot start requiring it after duplication of its neighbors,
    // so this set never has to grow later
    std::vector<VertId> vertsToProcess;
    for ( auto v = 0_v; v + 1 < all.vert2firstRec.size(); ++v )
        if ( all.vertInfos[v].duplicationNeeded() )
            vertsToProcess.push_back( v );

    auto sortPred = [&]( VertId a, VertId b )
    {
        const auto ai = all.vertInfos[a];
        const auto bi = all.vertInfos[b];
        if ( ai.hasRepeatedVerts() != bi.hasRepeatedVerts() )
            return bi.hasRepeatedVerts(); // process neighbourhoods without repeated vertices (a) first, because duplication of neighbours cannot help them

        if ( ai.hasRepeatedVerts() )
        {
            const int aTris = all.vert2firstRec[a+1] - all.vert2firstRec[a];
            const int bTris = all.vert2firstRec[b+1] - all.vert2firstRec[b];
            // double ring is the case when every triangle around central vertex is present in both orientations,
            // so every neighbour vertex is repeated
            const bool aTwinChains = ai.areTwinChains( aTris );
            const bool bTwinChains = bi.areTwinChains( bTris );
            if ( aTwinChains != bTwinChains )
                return aTwinChains; // process double ring vertices ahead of others, since their duplication produces two closed chains

            if ( ai.maxVertRepetitions() != bi.maxVertRepetitions() )
                return ai.maxVertRepetitions() < bi.maxVertRepetitions(); // process vertices with fewer maximal neighbour repetitions first

            // process neighbourhoods with more repeated vertices first,
            // keep normal (not reversed) order of the vertices with same number of neighbours' repetitions
            return std::make_pair( -(int)ai.numRepeatedVerts(), a ) < std::make_pair( -(int)bi.numRepeatedVerts(), b );
        }
        // process neighbourhoods with more chains first,
        // keep normal (not reversed) order of the vertices with same number of chains
        return std::make_pair( -(int)ai.numChains(), a ) < std::make_pair( -(int)bi.numChains(), b );
    };
    tbb::parallel_sort( vertsToProcess.begin(), vertsToProcess.end(), sortPred );

    PathAroundVertex pathMaker;
    std::vector<VertId> path;
    std::vector<VertId> closedPath;
    VertBitSet visitedVertices( all.recs.back().v ); // explicitly not `lastValidVert` but last vert used in triangulation
    size_t duplicatedVerticesCnt = 0;
    for ( auto v : vertsToProcess )
    {
        const auto posBegin = all.vert2firstRec[v];
        const auto posEnd = all.vert2firstRec[v + 1];

        // do not call inspector.run( t, all.recs.data() + posBegin, all.recs.data() + posEnd ).duplicationNeeded() to skip duplication,
        // because formal non-manifoldness of this vertex can be resolved by a neighbour vertex duplication,
        // but we still want to dupliate it to avoid neighbours with equal coordinates

        pathMaker.init( t, all.recs, posBegin, posEnd );

        // first chain of vertices around the center does not require duplication
        int foundChains = 0;
        while ( !pathMaker.empty() )
        {
            for(const auto& vi : path)
                visitedVertices.reset(vi);

            bool triOrientation = true;
            auto [firstVertex, nextVertex] = pathMaker.getFirstTwoVertices();
            visitedVertices.autoResizeSet( firstVertex );
            visitedVertices.autoResizeSet( nextVertex );
            VertId prevVertex = firstVertex;

            // preserve allocated memory in path
            path.clear();
            path.push_back( firstVertex );
            path.push_back( nextVertex );

            while ( true )
            {
                {
                    // prefer finding nextVertex not equal to prevVertex to maximize neighbour ring sizes
                    auto currVertex = nextVertex;
                    nextVertex = pathMaker.getNextVertex( currVertex, triOrientation, prevVertex, myDups );
                    prevVertex = currVertex;
                }

                if ( !nextVertex )
                {
                    if ( triOrientation ) // try the opposite direction from firstVertex
                    {
                        triOrientation = false;
                        prevVertex = path[1];
                        std::reverse( path.begin(), path.end() );
                        nextVertex = pathMaker.getNextVertex( firstVertex, triOrientation, prevVertex, myDups );
                        prevVertex = firstVertex;
                    }
                    if ( !nextVertex )
                    {
                        if ( foundChains )
                        {
                            pathMaker.duplicateVertex( v, path, lastValidVert, triOrientation, &myDups );
                            ++duplicatedVerticesCnt;
                        }
                        ++foundChains;
                        break;
                    }
                }

                // returned to already visited vertex
                if ( visitedVertices.test(nextVertex) )
                {
                    // save only closed path and prepare for new search starting with non-manifold vertex
                    path.push_back( nextVertex );
                    extractClosedPath( path, closedPath );
                    for( const auto& vi : closedPath)
                        visitedVertices.reset(vi);

                    if ( foundChains )
                    {
                        pathMaker.duplicateVertex( v, closedPath, lastValidVert, triOrientation, &myDups );
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

    if ( dups )
        *dups = std::move( myDups );

    return duplicatedVerticesCnt;
}

} //namespace MeshBuilder

} //namespace MR
