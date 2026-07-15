#pragma once

#include "MRId.h"
#include "MRPch/MRBindingMacros.h"
#include <utility>
#include <vector>

namespace MR
{

namespace MeshBuilder
{

struct VertDuplication
{
    VertId srcVert; ///< original vertex before duplication
    VertId dupVert; ///< new vertex after duplication
};

/// resolve non-manifold vertices by creating duplicate vertices in the triangulation (which is modified)
/// `lastValidVert` is needed if `region` or `t` does not contain full mesh, then first duplicated vertex will have `lastValidVert+1` index
/// return number of duplicated vertices
MRMESH_API size_t duplicateNonManifoldVertices( Triangulation & t, FaceBitSet * region = nullptr,
    std::vector<VertDuplication>* dups = nullptr, VertId lastValidVert = {} );

/// classification of the triangles around one vertex
struct VertInfo
{
    /// the number of chains of connected triangles around the vertex;
    /// numChains is set to 0 if numRepeatedVerts > 0
    int numChains = 0;

    /// the number of vertices, which are passed more than once
    int numRepeatedVerts = 0;
};

/// describes a vertex and one of triangles incident to it
struct VertTri
{
    VertId v;
    FaceId f;

    auto asPair() const { return std::make_pair( v, f ); }
    friend bool operator <( const VertTri& l, const VertTri& r ) { return l.asPair() < r.asPair(); }
};

/// computes VertInfo of one vertex given all its incident triangles in [begin, end), all referencing the same vertex;
/// allocates temporary hash maps on every call, so prefer batch processing over calling it per vertex of a large mesh
[[nodiscard]] MRMESH_API MR_BIND_IGNORE_PY VertInfo inspectVertNeighbourhood( const Triangulation & t, const VertTri * begin, const VertTri * end );

} //namespace MeshBuilder

} //namespace MR
