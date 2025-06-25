#pragma once

#include "MRId.h"
#include "MRMeshPart.h"
#include "MRPrecisePredicates3.h"

#include <functional>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/// edge from one mesh and triangle from another mesh
struct EdgeTri
{
    EdgeId edge;
    FaceId tri;
    EdgeTri() = default;
    EdgeTri( EdgeId e, FaceId t ) : edge( e ), tri( t ) { }
};

inline bool operator==( const EdgeTri& a, const EdgeTri& b )
{
    return a.edge.undirected() == b.edge.undirected() && a.tri == b.tri;
}

/// if isEdgeATriB() == true,  then stores edge from mesh A and triangle from mesh B
/// if isEdgeATriB() == false, then stores edge from mesh B and triangle from mesh A
struct VarEdgeTri
{
    EdgeId edge;
    struct FlaggedTri
    {
        unsigned int isEdgeATriB : 1 = 0;
        unsigned int face : 31 = 0;
        bool operator==( const FlaggedTri& ) const = default;
    } flaggedTri;

    [[nodiscard]] FaceId tri() const { return FaceId( flaggedTri.face ); }
    [[nodiscard]] bool isEdgeATriB() const { return bool( flaggedTri.isEdgeATriB ); }
    [[nodiscard]] EdgeTri edgeTri() const { return EdgeTri( edge, tri() ); }

    [[nodiscard]] bool valid() const { return edge.valid(); }
    [[nodiscard]] explicit operator bool() const { return edge.valid(); }

    VarEdgeTri() = default;
    VarEdgeTri( bool isEdgeATriB, EdgeId e, FaceId t )
    {
        assert( t.valid() );
        edge = e;
        flaggedTri.isEdgeATriB = isEdgeATriB;
        flaggedTri.face = t;
    }
    VarEdgeTri( bool isEdgeATriB, const EdgeTri& et ) : VarEdgeTri( isEdgeATriB, et.edge, et.tri ) {}

    [[nodiscard]] bool operator==( const VarEdgeTri& ) const = default;
};
static_assert( sizeof( VarEdgeTri ) == 8 );

/// In each VarEdgeTri = pair of (intersected edge, intersected triangle), the intersected edge
/// is directed from negative half-space of the intersected triangle B to its positive half-space
using PreciseCollisionResult = std::vector<VarEdgeTri>;

/**
 * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param anyIntersection if true then the function returns as fast as it finds any intersection
 */
MRMESH_API PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const MeshPart & b,
    ConvertToIntVector conv, const AffineXf3f* rigidB2A = nullptr, bool anyIntersection = false );

/**
 * \brief finds all pairs of colliding edges and triangle within one mesh
 * \param anyIntersection if true then the function returns as fast as it finds any intersection
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation, might be useful to obtain same result as in boolean operation would for mesh B
 * \param aVertsSize used in float to int conversion, might be useful to obtain same result as in boolean operation would for mesh B
 */
MRMESH_API std::vector<EdgeTri> findSelfCollidingEdgeTrisPrecise( const MeshPart& mp,
    ConvertToIntVector conv, bool anyIntersection = false, const AffineXf3f* rigidB2A = nullptr, int aVertSizes = 0 );

/// finds all intersections between every given edge from A and given triangles from B
MRMESH_API std::vector<EdgeTri> findCollidingEdgeTrisPrecise( 
    const Mesh & a, const std::vector<EdgeId> & edgesA,
    const Mesh & b, const std::vector<FaceId> & facesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A = nullptr );

/// finds all intersections between every given triangle from A and given edge from B
MRMESH_API std::vector<EdgeTri> findCollidingEdgeTrisPrecise( 
    const Mesh & a, const std::vector<FaceId> & facesA,
    const Mesh & b, const std::vector<EdgeId> & edgesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A = nullptr );

/**
 * \brief creates simple converters from Vector3f to Vector3i and back in mesh part area range
 */
MRMESH_API CoordinateConverters getVectorConverters( const MeshPart& a );

/**
 * \brief creates simple converters from Vector3f to Vector3i and back in mesh parts area range
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
MRMESH_API CoordinateConverters getVectorConverters( const MeshPart& a, const MeshPart& b,
    const AffineXf3f* rigidB2A = nullptr );

/// \}

} // namespace MR
