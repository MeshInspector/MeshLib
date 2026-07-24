#pragma once

#include "MRId.h"
#include "MRPch/MRBindingMacros.h"
#include <cassert>
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
/// `dups` (if given) contents will be ignored and overridden; it receives the duplications in creation order with consecutive dupVert ids starting from `lastValidVert+1`
/// return number of duplicated vertices
MRMESH_API size_t duplicateNonManifoldVertices( Triangulation & t, FaceBitSet * region = nullptr,
    std::vector<VertDuplication>* dups = nullptr, VertId lastValidVert = {} );

/// classification of the triangles around one vertex, packed in 32 bits;
/// it stores 1-bit flag (hasRepeatedVerts) and either 31-bit numRepeatedVerts (if the flag is on),
/// or 16-bit numOpenChains and 15-bit numClosedChains (if the flag is off)
struct VertInfo
{
    /// true if some neighbor vertex is present in more than one triangle-pair around the vertex
    [[nodiscard]] bool hasRepeatedVerts() const { return ( data_ & 1 ) != 0; }

    /// the number of neighbor vertex repetitions; 0 if !hasRepeatedVerts()
    [[nodiscard]] std::uint32_t numRepeatedVerts() const { return hasRepeatedVerts() ? data_ >> 1 : 0; }

    /// the number of open chains of connected triangles around the vertex; 0 if hasRepeatedVerts()
    [[nodiscard]] std::uint32_t numOpenChains() const { return hasRepeatedVerts() ? 0 : ( data_ >> 1 ) & maxNumOpenChains; }

    /// the number of closed chains (rings) of connected triangles around the vertex; 0 if hasRepeatedVerts()
    [[nodiscard]] std::uint32_t numClosedChains() const { return hasRepeatedVerts() ? 0 : data_ >> 17; }

    /// true if the triangles around the vertex do not form a configuration MeshBuilder accepts as is
    /// (a single chain or ring, no triangles at all, or two open chains), so the vertex must be duplicated
    [[nodiscard]] bool duplicationNeeded() const
    {
        return hasRepeatedVerts() ||
            !( numOpenChains() + numClosedChains() <= 1
            || ( numOpenChains() == 2 && numClosedChains() == 0 ) );
    }

    /// increments numRepeatedVerts saturating at its maximum; the first call zeros the chain counters forever
    void incRepeatedVerts()
    {
        if ( !hasRepeatedVerts() )
            data_ = 3; // the flag and numRepeatedVerts = 1
        else if ( numRepeatedVerts() < maxNumRepeatedVerts )
            data_ += 2;
    }

    /// increments numOpenChains saturating at its maximum
    void incOpenChains()
    {
        assert( !hasRepeatedVerts() );
        if ( numOpenChains() < maxNumOpenChains )
            data_ += 2;
    }

    /// decrements numOpenChains, but a saturated counter sticks to its maximum forever
    void decOpenChains()
    {
        assert( !hasRepeatedVerts() );
        assert( numOpenChains() > 0 );
        if ( numOpenChains() < maxNumOpenChains )
            data_ -= 2;
    }

    /// increments numClosedChains saturating at its maximum
    void incClosedChains()
    {
        assert( !hasRepeatedVerts() );
        if ( numClosedChains() < maxNumClosedChains )
            data_ += 1u << 17;
    }

    /// maximal values storable in the counters
    static constexpr std::uint32_t maxNumOpenChains = ( 1u << 16 ) - 1;
    static constexpr std::uint32_t maxNumClosedChains = ( 1u << 15 ) - 1;
    static constexpr std::uint32_t maxNumRepeatedVerts = ( 1u << 31 ) - 1;

private:
    std::uint32_t data_ = 0;
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
