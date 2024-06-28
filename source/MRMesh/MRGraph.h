#pragma once

#include "MRVector.h"
#include "MRId.h"
#include "MRBitSet.h"
#include <cassert>
#include <functional>

namespace MR
{

/// mathematical graph consisting from vertices and undirected edges
class Graph
{
public:
    using VertId = GraphVertId;
    using EdgeId = GraphEdgeId;

    using VertBitSet = GraphVertBitSet;
    using EdgeBitSet = GraphEdgeBitSet;

    using Neighbours = std::vector<EdgeId>; // sorted by edgeID
    using NeighboursPerVertex = Vector<Neighbours, VertId>;

    struct EndVertices
    {
        VertId v0, v1; // v0 < v1
        [[nodiscard]] VertId otherEnd( VertId a ) const
        {
            assert( a == v0 || a == v1 );
            return a == v0 ? v1 : v0;
        }
        void replaceEnd( VertId what, VertId with )
        {
            assert( what != with );
            assert( ( v0 == what && v1 != with ) || ( v1 == what && v0 != with ) );
            if ( v0 == what )
                v0 = with;
            else
                v1 = with;
            if ( v0 > v1 )
                std::swap( v0, v1 );
        }
        auto operator <=>( const EndVertices& ) const = default;
    };
    using EndsPerEdge = Vector<EndVertices, EdgeId>;

    /// constructs the graph from all valid vertices and edges
    MRMESH_API void construct( NeighboursPerVertex neighboursPerVertex, EndsPerEdge endsPerEdge );

    /// returns the number of vertex records, including invalid ones
    [[nodiscard]] size_t vertSize() const { return neighboursPerVertex_.size(); }

    /// returns all valid vertices in the graph
    [[nodiscard]] const VertBitSet & validVerts() const { return validVerts_; }

    /// returns true if given vertex is valid
    [[nodiscard]] bool valid( VertId v ) const { return validVerts_.test( v ); }

    /// returns the number of edge records, including invalid ones
    [[nodiscard]] size_t edgeSize() const { return endsPerEdge_.size(); }

    /// returns all valid edges in the graph
    [[nodiscard]] const EdgeBitSet & validEdges() const { return validEdges_; }

    /// returns true if given edge is valid
    [[nodiscard]] bool valid( EdgeId e ) const { return validEdges_.test( e ); }

    /// returns all edges adjacent to given vertex
    [[nodiscard]] const Neighbours & neighbours( VertId v ) const { return neighboursPerVertex_[v]; }

    /// returns the ends of given edge
    [[nodiscard]] const EndVertices & ends( EdgeId e ) const { return endsPerEdge_[e]; }

    /// finds and returns edge between vertices a and b; returns invalid edge otherwise
    [[nodiscard]] MRMESH_API EdgeId findEdge( VertId a, VertId b ) const;

    /// returns true if the vertices a and b are neighbors
    [[nodiscard]] bool areNeighbors( VertId a, VertId b ) const { return findEdge( a, b ).valid(); }

    /// unites two vertices into one (deleting the second one),
    /// which leads to deletion and merge of some edges, for which callback is called
    MRMESH_API void merge( VertId remnant, VertId dead, std::function<void( EdgeId remnant, EdgeId dead )> onMergeEdges );

    /// verifies that all internal data structures are valid
    MRMESH_API bool checkValidity() const;

private:
    VertBitSet validVerts_;
    EdgeBitSet validEdges_;

    NeighboursPerVertex neighboursPerVertex_;
    EndsPerEdge endsPerEdge_;
};

} //namespace MR
