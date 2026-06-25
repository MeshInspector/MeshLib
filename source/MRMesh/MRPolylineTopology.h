#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"
#include "MRId.h"
// for template implementation:
#include "MRBitSet.h"

namespace MR
{

/// topology of one or several polylines (how line segments are connected in lines) common for 2D and 3D polylines
/// \ingroup PolylineGroup
class PolylineTopology
{
public:
    /// builds this topology from given contours
    /// \details also builds the vector of referenced points using two functors: reserve and add \n
    /// if all even edges are consistently oriented, then the output contours will be oriented the same
    template<typename T, typename F1, typename F2>
    void buildFromContours( const std::vector<std::vector<T>> & contours, F1 && reservePoints, F2 && addPoint );

    /// build topology of comp2firstVert.size()-1 not-closed polylines
    /// each pair (a,b) of indices in \param comp2firstVert defines vertex range of a polyline [a,b)
    MRMESH_API void buildOpenLines( const std::vector<VertId> & comp2firstVert );

    /// converts this topology into contours of given type using the functor returning point by its Id
    template<typename T, typename F>
    [[nodiscard]] std::vector<std::vector<T>> convertToContours( F&& getPoint, std::vector<std::vector<VertId>>* vertMap = nullptr ) const;

    /// creates an edge not associated with any vertex
    [[nodiscard]] MRMESH_API EdgeId makeEdge();

    /// makes an edge connecting vertices a and b
    /// \param e if valid then the function does not make new edge, but connects the vertices using given one (and returns it)
    /// \details if
    ///   1) any of the vertices is invalid or not within the vertSize(),
    ///   2) or a==b,
    ///   3) or either a or b already has 2 incident edges,
    ///   4) given edge e is not lone or not within the edgeSize()
    /// then makeEdge(a,b) does nothing and returns invalid edge
    MRMESH_API EdgeId makeEdge( VertId a, VertId b, EdgeId e = {} );

    /// for every given edge[ue] calls makeEdge( edge[ue][0], edge[ue][1], ue ), making new edges so that edges.size() <= undirectedEdgeSize() at the end
    /// \return the total number of edges created
    MRMESH_API int makeEdges( const Edges & edges );

    /// checks whether the edge is disconnected from all other edges and disassociated from all vertices (as if after makeEdge)
    [[nodiscard]] MRMESH_API bool isLoneEdge( EdgeId a ) const;

    /// returns last not lone undirected edge id, or invalid id if no such edge exists
    [[nodiscard]] MRMESH_API UndirectedEdgeId lastNotLoneUndirectedEdge() const;

    /// returns last not lone edge id, or invalid id if no such edge exists
    [[nodiscard]] EdgeId lastNotLoneEdge() const { auto ue = lastNotLoneUndirectedEdge(); return ue ? EdgeId( ue ) + 1 : EdgeId(); }

    /// returns the number of half-edge records including lone ones
    [[nodiscard]] size_t edgeSize() const { return edges_.size(); }

    /// returns the number of allocated edge records
    [[nodiscard]] size_t edgeCapacity() const { return edges_.capacity(); }

    /// returns the number of undirected edges (pairs of half-edges) including lone ones
    [[nodiscard]] size_t undirectedEdgeSize() const { return edges_.size() >> 1; }

    /// returns the number of allocated undirected edges (pairs of half-edges)
    [[nodiscard]] size_t undirectedEdgeCapacity() const { return edges_.capacity() >> 1; }

    /// computes the number of not-lone (valid) undirected edges
    [[nodiscard]] MRMESH_API size_t computeNotLoneUndirectedEdges() const;

    /// sets the capacity of half-edges vector
    void edgeReserve( size_t newCapacity ) { edges_.reserve( newCapacity ); }

    /// returns true if given edge is within valid range and not-lone
    [[nodiscard]] bool hasEdge( EdgeId e ) const { assert( e.valid() ); return e < (int)edgeSize() && !isLoneEdge( e ); }

    /// given edge becomes lone after the call, so it is un-spliced from connected edges, and if it was not connected at origin or destination, then that vertex is deleted as well
    MRMESH_API void deleteEdge( UndirectedEdgeId ue );

    /// calls deleteEdge for every set bit
    MRMESH_API void deleteEdges( const UndirectedEdgeBitSet & es );

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

    /// given two half edges do either of two: \n
    /// 1) if a and b were from distinct rings, puts them in one ring; \n
    /// 2) if a and b were from the same ring, puts them in separate rings;
    /// \details the cut in rings in both cases is made after a and b
    MRMESH_API void splice( EdgeId a, EdgeId b );
    
    /// next (counter clock wise) half-edge in the origin ring
    [[nodiscard]] EdgeId next( EdgeId he ) const { assert(he.valid()); return edges_[he].next; }

    /// returns origin vertex of half-edge
    [[nodiscard]] VertId org( EdgeId he ) const { assert(he.valid()); return edges_[he].org; }

    /// returns destination vertex of half-edge
    [[nodiscard]] VertId dest( EdgeId he ) const { assert(he.valid()); return edges_[he.sym()].org; }

    /// sets new origin to the full origin ring including this edge
    /// \details edgePerVertex_ table is updated accordingly
    MRMESH_API void setOrg( EdgeId a, VertId v );

    /// for all valid vertices this vector contains an edge with the origin there
    [[nodiscard]] const Vector<EdgeId, VertId> & edgePerVertex() const { return edgePerVertex_; }

    /// for all edges this vector contains its origin
    [[nodiscard]] MRMESH_API Vector<VertId, EdgeId> getOrgs() const;

    /// returns valid edge if given vertex is present in the mesh
    [[nodiscard]] EdgeId edgeWithOrg( VertId a ) const { assert( a.valid() ); return a < int(edgePerVertex_.size()) && edgePerVertex_[a].valid() ? edgePerVertex_[a] : EdgeId(); }

    /// returns true if given vertex is present in the mesh
    [[nodiscard]] bool hasVert( VertId a ) const { return validVerts_.test( a ); }

    /// returns 0 if given vertex does not exist, 1 if it has one incident edge, and 2 if it has two incident edges
    [[nodiscard]] MRMESH_API int getVertDegree( VertId a ) const;

    /// returns the number of valid vertices
    [[nodiscard]] int numValidVerts() const { return numValidVerts_; }

    /// returns last valid vertex id, or invalid id if no single valid vertex exists
    [[nodiscard]] MRMESH_API VertId lastValidVert() const;

    /// creates new vert-id not associated with any edge yet
    [[nodiscard]] VertId addVertId() { edgePerVertex_.push_back( {} ); validVerts_.push_back( false ); return VertId( (int)edgePerVertex_.size() - 1 ); }

    /// explicitly increases the size of vertices vector
    MRMESH_API void vertResize( size_t newSize );

    /// explicitly increases the size of vertices vector, doubling the current capacity if it was not enough
    MRMESH_API void vertResizeWithReserve( size_t newSize );

    /// sets the capacity of vertices vector
    void vertReserve( size_t newCapacity ) { edgePerVertex_.reserve( newCapacity ); validVerts_.reserve( newCapacity ); }

    /// returns the number of vertex records including invalid ones
    [[nodiscard]] size_t vertSize() const { return edgePerVertex_.size(); }

    /// returns the number of allocated vert records
    [[nodiscard]] size_t vertCapacity() const { return edgePerVertex_.capacity(); }

     /// returns cached set of all valid vertices
    [[nodiscard]] const VertBitSet & getValidVerts() const { return validVerts_; }

    /// if region pointer is not null then converts it in reference, otherwise returns all valid vertices in the mesh
    [[nodiscard]] const VertBitSet & getVertIds( const VertBitSet * region ) const { return region ? *region : validVerts_; }

    /// finds and returns edge from o to d in the mesh; returns invalid edge otherwise
    [[nodiscard]] MRMESH_API EdgeId findEdge( VertId o, VertId d ) const;

    /// returns all vertices incident to path edges
    [[nodiscard]] MRMESH_API VertBitSet getPathVertices( const EdgePath & path ) const;

    /// split given edge on two parts:
    /// dest(returned-edge) = org(e) - newly created vertex,
    /// org(returned-edge) = org(e-before-split),
    /// dest(e) = dest(e-before-split)
    MRMESH_API EdgeId splitEdge( EdgeId e );

    /// adds polyline in this topology passing progressively via vertices *[vs, vs+num);
    /// if vs[0] == vs[num-1] then a closed polyline is created;
    /// return the edge from first to second vertex
    MRMESH_API EdgeId makePolyline( const VertId * vs, size_t num );

    /// appends polyline topology (from) in addition to the current topology: creates new edges, verts;
    /// \param outVmap,outEmap (optionally) returns mappings: from.id -> this.id
    MRMESH_API void addPart( const PolylineTopology & from,
        VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr );

    /// appends polyline topology (from) in addition to the current topology: creates new edges, verts;
    MRMESH_API void addPartByMask( const PolylineTopology& from, const UndirectedEdgeBitSet& mask,
        VertMap* outVmap = nullptr, EdgeMap* outEmap = nullptr );

    /// tightly packs all arrays eliminating lone edges and invalid vertices
    /// \param outVmap,outEmap if given returns mappings: old.id -> new.id;
    MRMESH_API void pack( VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr );

    /// saves this in binary stream
    MRMESH_API void write( std::ostream & s ) const;

    /// loads this from binary stream
    MRMESH_API bool read( std::istream & s );

    /// comparison via edges (all other members are considered as not important caches)
    [[nodiscard]] bool operator ==( const PolylineTopology & b ) const { return edges_ == b.edges_; }
    [[nodiscard]] bool operator !=( const PolylineTopology & b ) const { return edges_ != b.edges_; }

    /// returns true if for each edge e: e == e.next() || e.odd() == next( e ).sym().odd()
    [[nodiscard]] MRMESH_API bool isConsistentlyOriented() const;

    /// changes the orientation of all edges: every edge e is replaced with e.sym()
    MRMESH_API void flip();

    /// verifies that all internal data structures are valid
    MRMESH_API bool checkValidity() const;

    /// computes numValidVerts_ and validVerts_ from edgePerVertex_
    MRMESH_API void computeValidsFromEdges();

    /// returns true if the polyline does not have any holes
    [[nodiscard]] MRMESH_API bool isClosed() const;

private:
    /// sets new origin to the full origin ring including this edge, without updating edgePerVertex_ table
    void setOrg_( EdgeId a, VertId v );

    /// data of every half-edge
    struct HalfEdgeRecord
    {
        EdgeId next; ///< next counter clock wise half-edge in the origin ring
        VertId org;  ///< vertex at the origin of the edge

        bool operator ==( const HalfEdgeRecord& b ) const
        {
            return next == b.next && org == b.org;
        }
        HalfEdgeRecord() noexcept = default;
        explicit HalfEdgeRecord( NoInit ) noexcept : next( noInit ), org( noInit ) {}
    };

    /// edges_: EdgeId -> edge data
    Vector<HalfEdgeRecord, EdgeId> edges_;

    /// edgePerVertex_: VertId -> one edge id of one of edges with origin there
    Vector<EdgeId, VertId> edgePerVertex_;
    VertBitSet validVerts_; ///< each true bit here corresponds to valid element in edgePerVertex_
    int numValidVerts_ = 0; ///< the number of valid elements in edgePerVertex_ or set bits in validVerts_
};

template<typename T, typename F1, typename F2>
void PolylineTopology::buildFromContours( const std::vector<std::vector<T>> & contours, F1 && reservePoints, F2 && addPoint )
{
    *this = {};
    size_t size = 0;
    std::vector<bool> closed;
    closed.reserve( contours.size() );
    int numClosed = 0;
    for ( const auto& c : contours )
    {
        const auto csize = c.size();
        if ( csize > 2 )
        {
            closed.push_back( c.front() == c.back() );
        }
        else
        {
            closed.push_back( false );
        }
        if ( csize < 2 )
            continue; // ignore contours with 0 or 1 points because of no edges in them
        size += c.size();
        if ( closed.back() )
            ++numClosed;
    }

    reservePoints( size - numClosed );
    vertResize( size - numClosed );

    for ( int i = 0; i < contours.size(); ++i )
    {
        const auto& c = contours[i];
        if ( c.size() < 2 )
            continue; // ignore contours with 0 or 1 points because of no edges in them
        const auto e0 = makeEdge();
        const auto v0 = addPoint( c[0] );
        setOrg( e0, v0 );
        auto e = e0;
        for ( int j = 1; j + 1 < c.size(); ++j )
            {
            const auto ej = makeEdge();
            splice( ej, e.sym() );
            const auto vj = addPoint( c[j] );
            setOrg( ej, vj );
            e = ej;
        }
        if ( closed[i] )
        {
            splice( e0, e.sym() );
        }
        else
        {
            const auto vx = addPoint( c.back() );
            setOrg( e.sym(), vx );
        }
    }
    assert( isConsistentlyOriented() );
    assert( edgePerVertex_.size() ==  size - numClosed );
}

template<typename T, typename F>
std::vector<std::vector<T>> PolylineTopology::convertToContours( F&& getPoint, std::vector<std::vector<VertId>>* vertMap ) const
{
    std::vector<std::vector<T>> res;

    UndirectedEdgeBitSet linesUsed;
    linesUsed.autoResizeSet( UndirectedEdgeId{ undirectedEdgeSize() } );
    linesUsed.flip();
    for ( EdgeId e0 : linesUsed )
    {
        if ( isLoneEdge( e0 ) )
            continue;

        EdgeId curLine = e0;
        while ( curLine != next( curLine ) ) /// until contour start is reached
        {
            curLine = next( curLine ).sym();
            if ( curLine == e0 )
                break; /// the contour is closed
        }

        linesUsed.set( curLine.undirected(), false );

        EdgeId e = curLine;
        std::vector<T> cont;
        std::vector<VertId> map;
        auto orgV = org( e );
        cont.push_back( getPoint( orgV ) );
        if ( vertMap )
            map.push_back( orgV );
        for ( ;; )
        {
            e = e.sym();
            cont.push_back( getPoint( org( e ) ) );
            e = next( e );
            if ( !linesUsed.test_set( e.undirected(), false ) )
                break;
        }
        res.push_back( std::move( cont ) );
        if ( vertMap )
            vertMap->push_back( std::move( map ) );
    }

    return res;
}

/// simplifies construction of connected polyline in the topology
struct PolylineMaker
{
    PolylineTopology & topology;
    PolylineMaker( PolylineTopology & t ) : topology( t ) {}

    /// creates first edge of polyline
    /// \param v first vertex of the polyline
    EdgeId start( VertId v )
    {
        assert( !e0_ && !eLast_ );
        e0_ = eLast_ = topology.makeEdge();
        topology.setOrg( e0_, v );
        return e0_;
    }
    /// makes next edge of polyline
    /// \param v next vertex of the polyline
    EdgeId proceed( VertId v )
    {
        assert( eLast_ );
        const auto ej = topology.makeEdge();
        topology.splice( ej, eLast_.sym() );
        topology.setOrg( ej, v );
        return eLast_ = ej;
    }
    /// closes the polyline
    void close()
    {
        assert( e0_ && eLast_ );
        topology.splice( e0_, eLast_.sym() );
        e0_ = eLast_ = {};
    }
    /// finishes the polyline adding final vertex in it
    void finishOpen( VertId v )
    {
        assert( eLast_ );
        topology.setOrg( eLast_.sym(), v );
        e0_ = eLast_ = {};
    }

private:
    EdgeId e0_, eLast_;
};

} // namespace MR
