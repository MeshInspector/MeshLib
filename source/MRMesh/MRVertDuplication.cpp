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
    Triangulation& faceToVertices;
    // all iterators in [vertexBegIt, vertexEndIt) must have the same central vertex
    std::vector<VertTri>::iterator vertexBegIt, vertexEndIt;
    size_t firstUnvisitedIndex = 0; // pivot index. [vertexBegIt + firstUnvistedIndex, vertexBegIt) - unvisited vertices

public:
    PathAroundVertex( Triangulation& triangleToVertices,
                std::vector<VertTri>& vertTris, size_t beg, size_t end )
        : faceToVertices( triangleToVertices )
        , vertexBegIt( vertTris.begin() + beg )
        , vertexEndIt( vertTris.begin() + end )
    {}

    // false if there are some unvisited vertices
    bool empty() const
    {
        return vertexBegIt + firstUnvisitedIndex >= vertexEndIt;
    }

    // first unvisited vertex
    VertId getFirstVertex() const
    {
        assert( !empty() );
        const auto first = vertexBegIt + firstUnvisitedIndex;
        // below selection ensures that getNextIncidentVertex( getFirstVertex(), true ) will find nextVertex in the very first triangle
        const auto & vs = faceToVertices[first->f];
        return getOtherTriVerts( vs, first->v ).first;
    }

    // find incident unvisited vertex, in case of several option prefer finding the vertex not equal to preVertex
    VertId getNextVertex( VertId v, bool triOrientation, VertId prevVertex = {} )
    {
        if ( empty() )
            return VertId( -1 );

        auto prevIt = vertexEndIt;
        for ( auto it = vertexBegIt + firstUnvisitedIndex; it < vertexEndIt; ++it )
        {
            VertId nextVertex;
            const auto & vs = faceToVertices[it->f];
            const auto v12 = getOtherTriVerts( vs, it->v );
            if ( triOrientation && v12.first == v )
                nextVertex = v12.second;
            else if ( !triOrientation && v12.second == v )
                nextVertex = v12.first;
            if ( nextVertex )
            {
                if ( nextVertex != prevVertex )
                {
                    if ( it != vertexBegIt + firstUnvisitedIndex )
                        std::iter_swap( it, vertexBegIt + firstUnvisitedIndex );
                    ++firstUnvisitedIndex;
                    return nextVertex;
                }
                // prevVertex is a possible continuation, store it, and search for other options
                prevIt = it;
            }
        }
        if ( prevIt < vertexEndIt )
        {
            // the only option is return in prevVertex
            if ( prevIt != vertexBegIt + firstUnvisitedIndex )
                std::iter_swap( prevIt, vertexBegIt + firstUnvisitedIndex );
            ++firstUnvisitedIndex;
            return prevVertex;
        }
        return {};
    }

    // duplicate the vertex around which the chain was found
    void duplicateVertex( VertId v, const std::vector<VertId>& path, VertId& lastUsedVertId, bool triOrientation,
                          std::vector<VertDuplication>* dups = nullptr )
    {
        VertDuplication vertDup;
        vertDup.dupVert = ++lastUsedVertId;
        vertDup.srcVert = v;
        if ( dups )
            dups->push_back( vertDup );

        [[maybe_unused]] size_t changedTris = 0;
        for ( size_t i = 1; i < path.size(); ++i )
        {
            for ( auto it = vertexBegIt; it < vertexBegIt + firstUnvisitedIndex; ++it )
            {
                VertId v1, v2;
                bool alreadyDuplicted = true;
                for ( VertId vi : faceToVertices[it->f] )
                {
                    if ( vi == vertDup.srcVert )
                    {
                        alreadyDuplicted = false;
                        // make (v1,v2) the cyclic pair following srcVert in the triangle
                        if ( v1 && !v2 )
                            std::swap( v1, v2 );
                    }
                    else if ( !v1 )
                        v1 = vi;
                    else if ( !v2 )
                        v2 = vi;
                }
                if ( alreadyDuplicted )
                    continue;
                assert( v1 && v2 );
                assert( v1 != v2 );

                if ( ( triOrientation && v1 == path[i - 1] && v2 == path[i] ) ||
                     ( !triOrientation && v2 == path[i - 1] && v1 == path[i] ) )
                {
                    for ( VertId & vi : faceToVertices[it->f] )
                    {
                        if ( vi != vertDup.srcVert )
                            continue;
                        vi = vertDup.dupVert;
                        break;
                    }
                    ++changedTris;
                    it->v = vertDup.dupVert;
                    break;
                }
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
    /// l_[v1] is present in the map, if there is a triangle to the left of (v,v1) edge;
    /// l_[v1]'s value is invalid if there is a triangle to the right of (v,v1) edge;
    /// otherwise it is the vertex v2 such that there is a chain of triangles in between (v,v1) and (v,v2) and there is no triangle to the left of (v,v2) edge
    HashMap<VertId, VertId> l_;

    /// r_[v2] is present in the map, if there is a triangle to the right of (v,v2) edge;
    /// r_[v2]'s value is invalid if there is a triangle to the left of (v,v2) edge;
    /// otherwise it is the vertex v1 such that there is a chain of triangles in between (v,v1) and (v,v2) and there is no triangle to the right of (v,v1) edge
    HashMap<VertId, VertId> r_;
};

VertInfo inspectVertNeighbourhood( const Triangulation & t, const VertTri * begin, const VertTri * end )
{
    return VertNeighbourhoodInspector{}.run( t, begin, end );
}

VertInfo VertNeighbourhoodInspector::run( const Triangulation & t, const VertTri * begin, const VertTri * end )
{
    l_.clear();
    r_.clear();
    VertInfo info;
    if ( begin == end )
        return info;
    const auto v0 = begin->v;
    for ( auto i = begin; i != end; ++i )
    {
        assert( i->v == v0 );
        const auto [v1, v2] = getOtherTriVerts( t[i->f], v0 );
        const auto lInsertion = l_.insert( { v1, v2 } );
        const auto rInsertion = r_.insert( { v2, v1 } );
        if ( !info.hasRepeatedVerts() && lInsertion.second && rInsertion.second )
        {
            info.incOpenChains();
            if ( auto it = l_.find( v2 ); it != l_.end() )
            {
                // the edge (v,v2) becomes inner
                const auto vEnd = it->second;
                it->second = VertId{};
                assert( vEnd ); // the edge (v,v2) was boundary
                info.decOpenChains();
                lInsertion.first->second = vEnd;
                assert( r_[vEnd] == v2 );
                r_[vEnd] = v1;
            }
            if ( auto it = r_.find( v1 ); it != r_.end() )
            {
                // the edge (v,v1) becomes inner
                const auto vEnd = it->second;
                it->second = VertId{};
                assert( vEnd ); // the edge (v,v1) was boundary
                if ( vEnd == v1 )
                {
                    // the chain is closed
                    assert( lInsertion.first->second == v1 );
                    lInsertion.first->second = VertId{};
                    rInsertion.first->second = VertId{};
                    info.decOpenChains();
                    info.incClosedChains();
                }
                else
                {
                    info.decOpenChains();
                    // the right end of the chain grown from the current triangle: v2, or updated by the merge above
                    const auto vRight = lInsertion.first->second;
                    assert( vRight );
                    if ( vRight != v2 )
                    {
                        // the current triangle merged two chains on both sides, so its both edges are inner
                        lInsertion.first->second = VertId{};
                        rInsertion.first->second = VertId{};
                        assert( r_[vRight] == v1 );
                        r_[vRight] = vEnd;
                    }
                    else
                        rInsertion.first->second = vEnd;
                    assert( l_[vEnd] == v1 );
                    l_[vEnd] = vRight;
                }
            }
        }
        else
        {
            // insertion can fail only if the vertex is repeated
            if ( !lInsertion.second )
                info.incRepeatedVerts();
            if ( !rInsertion.second )
                info.incRepeatedVerts();
        }
    }
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

/// returns true if the vertex with such neighborhood does not require duplication:
/// a single chain of triangles (or no triangles at all), or two open chains, which MeshBuilder has no issue with
static bool noDuplicationNeeded( const VertInfo & vertInfo )
{
    return !vertInfo.hasRepeatedVerts() &&
        ( vertInfo.numOpenChains() + vertInfo.numClosedChains() <= 1
        || ( vertInfo.numOpenChains() == 2 && vertInfo.numClosedChains() == 0 ) );
}

// for all vertices get over all incident vertices to find connected sequences
size_t duplicateNonManifoldVertices( Triangulation & t, FaceBitSet * region, std::vector<VertDuplication>* dups, VertId lastValidVert )
{
    MR_TIMER;
    if ( t.empty() )
        return 0; // input triangulation is empty

    AllVertTris all( t, region );
    if ( all.recs.empty() )
        return 0; // input triangulation contains only degenerate triangles, e.g. with repeating vertex (v v u)

    if ( !lastValidVert )
        lastValidVert = all.recs.back().v;

    all.computeVertSpans();
    all.computeVertInfos( t );

    std::vector<VertId> voi;
    for ( auto v = 0_v; v + 1 < all.vert2firstRec.size(); ++v )
        if ( !noDuplicationNeeded( all.vertInfos[v] ) )
            voi.push_back( v );

    auto sortPred = [&]( VertId a, VertId b )
    {
        const auto ai = all.vertInfos[a];
        const auto bi = all.vertInfos[b];
        if ( ai.hasRepeatedVerts() != bi.hasRepeatedVerts() )
            return bi.hasRepeatedVerts(); // process neighbourhoods without repeated vertices (a) first, because duplication of neighbours cannot help them
        if ( ai.hasRepeatedVerts() ) // process neighbourhoods with more repeated vertices (a) first
            return std::make_pair( ai.numRepeatedVerts(), a ) > std::make_pair( bi.numRepeatedVerts(), b );
        // process neighbourhoods with more chains (a) first
        return std::make_pair( ai.numOpenChains() + ai.numClosedChains(), a ) > std::make_pair( bi.numOpenChains() + bi.numClosedChains(), b );
    };
    tbb::parallel_sort( voi.begin(), voi.end(), sortPred );

    VertNeighbourhoodInspector inspector;
    std::vector<VertId> path;
    std::vector<VertId> closedPath;
    VertBitSet visitedVertices( all.recs.back().v ); // explicitly not `lastValidVert` but last vert used in triangulation
    size_t duplicatedVerticesCnt = 0;
    for ( auto v : voi )
    {
        // skip a vertex based on the neighborhood in the original triangulation;
        // a vertex not requiring duplication cannot start requiring it after duplication of its neighbors
        if ( noDuplicationNeeded( all.vertInfos[v] ) )
            continue;
        const auto posBegin = all.vert2firstRec[v];
        const auto posEnd = all.vert2firstRec[v + 1];
        // duplication of one vertex can resolve non-manifoldness in its neighbor vertex,
        // so after the first duplication recheck the neighborhood in the current triangulation
        if ( duplicatedVerticesCnt > 0 && noDuplicationNeeded( inspector.run( t, all.recs.data() + posBegin, all.recs.data() + posEnd ) ) )
            continue;
        PathAroundVertex pathMaker( t, all.recs, posBegin, posEnd );

        // first chain of vertices around the center does not require duplication
        int foundChains = 0;
        while ( !pathMaker.empty() )
        {
            for(const auto& vi : path)
                visitedVertices.reset(vi);

            bool triOrientation = true;
            const VertId firstVertex = pathMaker.getFirstVertex();
            visitedVertices.autoResizeSet( firstVertex );
            VertId prevVertex = firstVertex;
            VertId nextVertex = pathMaker.getNextVertex( firstVertex, triOrientation );
            if ( !nextVertex )
            {
                triOrientation = false;
                nextVertex = pathMaker.getNextVertex( firstVertex, triOrientation );
                assert( nextVertex.valid() );
            }
            visitedVertices.autoResizeSet( nextVertex );

            // preserve allocated memory in path
            path.clear();
            path.push_back( firstVertex );
            path.push_back( nextVertex );

            while ( true )
            {
                {
                    // prefer finding nextVertex not equal to prevVertex to maximize neighbour ring sizes
                    auto currVertex = nextVertex;
                    nextVertex = pathMaker.getNextVertex( currVertex, triOrientation, prevVertex );
                    prevVertex = currVertex;
                }

                if ( !nextVertex )
                {
                    if ( triOrientation ) // try the opposite direction from firstVertex
                    {
                        triOrientation = false;
                        prevVertex = path[1];
                        std::reverse( path.begin(), path.end() );
                        nextVertex = pathMaker.getNextVertex( firstVertex, triOrientation, prevVertex );
                    }
                    if ( !nextVertex )
                    {
                        if ( foundChains )
                        {
                            pathMaker.duplicateVertex( v, path, lastValidVert, triOrientation, dups );
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
                        pathMaker.duplicateVertex( v, closedPath, lastValidVert, triOrientation, dups );
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

} //namespace MeshBuilder

} //namespace MR
