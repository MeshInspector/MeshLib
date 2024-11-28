#pragma once

#include "MRFaceFace.h"
#include "MRMeshPart.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/**
 * \brief finds all pairs of colliding triangles from two meshes or two mesh regions
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 * \param firstIntersectionOnly if true then the function returns at most one pair of intersecting triangles and returns faster
 */
[[nodiscard]] MRMESH_API std::vector<FaceFace> findCollidingTriangles( const MeshPart & a, const MeshPart & b, 
    const AffineXf3f * rigidB2A = nullptr, bool firstIntersectionOnly = false );

/// the same as \ref findCollidingTriangles, but returns one bite set per mesh with colliding triangles
[[nodiscard]] MRMESH_API std::pair<FaceBitSet, FaceBitSet> findCollidingTriangleBitsets( const MeshPart& a, const MeshPart& b,
    const AffineXf3f* rigidB2A = nullptr );

/// finds all pairs (or the fact of any self-collision) of colliding triangles from one mesh or a region
[[nodiscard]] MRMESH_API Expected<bool> findSelfCollidingTriangles( const MeshPart& mp,
    std::vector<FaceFace> * outCollidingPairs, ///< if nullptr then the algorithm returns with true as soon as first collision is found
    ProgressCallback cb = {},
    const Face2RegionMap * regionMap = nullptr ); ///< if regionMap is provided then only self-intersections within a region are returned

/// finds all pairs of colliding triangles from one mesh or a region
[[nodiscard]] MRMESH_API Expected<std::vector<FaceFace>> findSelfCollidingTriangles( const MeshPart& mp, ProgressCallback cb = {},
    const Face2RegionMap * regionMap = nullptr ); ///< if regionMap is provided then only self-intersections within a region are returned

/// the same \ref findSelfCollidingTriangles but returns the union of all self-intersecting faces
[[nodiscard]] MRMESH_API Expected<FaceBitSet> findSelfCollidingTrianglesBS( const MeshPart & mp, ProgressCallback cb = {},
    const Face2RegionMap * regionMap = nullptr ); ///< if regionMap is provided then only self-intersections within a region are returned
 
/**
 * \brief checks that arbitrary mesh part A is inside of closed mesh part B
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
[[nodiscard]] MRMESH_API bool isInside( const MeshPart & a, const MeshPart & b, const AffineXf3f * rigidB2A = nullptr );

/**
 * \brief checks that arbitrary mesh part A is inside of closed mesh part B
 * The version of `isInside` without collision check; it is user's responsibility to guarantee that the meshes don't collide
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
[[nodiscard]] MRMESH_API bool isNonIntersectingInside( const MeshPart & a, const MeshPart & b, const AffineXf3f * rigidB2A = nullptr );

/**
 * \brief checks that arbitrary mesh A part (whole part is represented by one face `partFace`) is inside of closed mesh part B
 * The version of `isInside` without collision check; it is user's responsibility to guarantee that the meshes don't collide
 * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
 */
[[nodiscard]] MRMESH_API bool isNonIntersectingInside( const Mesh& a, FaceId partFace, const MeshPart& b, const AffineXf3f* rigidB2A = nullptr );

/// \}

} // namespace MR
