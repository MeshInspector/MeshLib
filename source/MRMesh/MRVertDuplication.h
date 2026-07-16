#pragma once

#include "MRId.h"
#include "MRPch/MRBindingMacros.h"
#include <cstdint>
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

/// classification of the triangles around one vertex, packed in 32 bits;
/// each counter saturates at its maximum value instead of overflowing
struct VertInfo
{
    /// the number of open chains of connected triangles around the vertex;
    /// set to 0 if numRepeatedVerts > 0
    std::uint32_t numOpenChains : 10 = 0;

    /// the number of closed chains (rings) of connected triangles around the vertex;
    /// set to 0 if numRepeatedVerts > 0
    std::uint32_t numClosedChains : 10 = 0;

    /// the number of vertices, which are passed more than once
    std::uint32_t numRepeatedVerts : 12 = 0;

    /// maximal values storable in the bit-fields above
    static constexpr std::uint32_t maxNumOpenChains = 1023;
    static constexpr std::uint32_t maxNumClosedChains = 1023;
    static constexpr std::uint32_t maxNumRepeatedVerts = 4095;
};
static_assert( sizeof( VertInfo ) == 4 );

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
