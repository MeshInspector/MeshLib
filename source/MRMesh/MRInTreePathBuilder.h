#pragma once

#include "MRVector.h"

namespace MR
{

/// given a spanning tree of edges in the mesh (or forest in case of several connected components),
/// prepares to build quickly a path along tree edges between any two vertices
class InTreePathBuilder
{
public:
    MRMESH_API InTreePathBuilder( const MeshTopology & topology, const UndirectedEdgeBitSet & treeEdges );

    /// finds the path in tree from start vertex to finish vertex
    [[nodiscard]] MRMESH_API EdgePath build( VertId start, VertId finish ) const;

private:
    /// given a vertex, returns the edge from it toward the root
    /// or invalid edge if v is a root
    [[nodiscard]] EdgeId getEdgeBack_( VertId v ) const;

    const MeshTopology & topology_;
    const UndirectedEdgeBitSet & treeEdges_;
    /// distance of each vertex from tree root in edges, and -1 for unreachable vertices
    Vector<int, VertId> vertDistance_;
};

} //namespace MR
