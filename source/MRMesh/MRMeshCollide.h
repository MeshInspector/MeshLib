#pragma once

#include "MRFaceFace.h"
#include "MRMeshPart.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/**
 * \brief finds all pairs of colliding triangles from two meshes or two mesh regions
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param firstIntersectionOnly if true then the function returns at most one pair of intersecting triangles and returns faster
 */
MRMESH_API std::vector<FaceFace> findCollidingTriangles( const MeshPart & a, const MeshPart & b, 
    const AffineXf3f * rigidB2A = nullptr, bool firstIntersectionOnly = false );

/// the same as \ref findCollidingTriangles, but returns one bite set per mesh with colliding triangles
MRMESH_API std::pair<FaceBitSet, FaceBitSet> findCollidingTriangleBitsets( const MeshPart& a, const MeshPart& b,
    const AffineXf3f* rigidB2A = nullptr );

/// finds all pairs of colliding triangles from one mesh or a region
MRMESH_API std::vector<FaceFace> findSelfCollidingTriangles( const MeshPart & mp );
/// the same \ref findSelfCollidingTriangles but returns the union of all self-intersecting faces
MRMESH_API FaceBitSet findSelfCollidingTrianglesBS( const MeshPart & mp );
 
/**
 * \brief checks that arbitrary mesh part A is inside of closed mesh part B
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
MRMESH_API bool isInside( const MeshPart & a, const MeshPart & b, const AffineXf3f * rigidB2A = nullptr );

/**
 * \brief checks that arbitrary mesh part A is inside of closed mesh part B
 * The version of `isInside` without collision check; it is user's responsibility to guarantee that the meshes don't collide
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
MRMESH_API bool isNonIntersectingInside( const MeshPart & a, const MeshPart & b, const AffineXf3f * rigidB2A = nullptr );

/// \}

} // namespace MR
