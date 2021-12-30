#pragma once

#include "MRFaceFace.h"
#include "MRMeshPart.h"

namespace MR
{

// finds all pairs of colliding triangles from two meshes or two mesh regions
MRMESH_API std::vector<FaceFace> findCollidingTriangles( const MeshPart & a, const MeshPart & b, 
    const AffineXf3f * rigidB2A = nullptr, // rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    bool firstIntersectionOnly = false );  // if true then the function returns at most one pair of intersecting triangles and returns faster
// the same, but returns one bite set per mesh with colliding triangles
MRMESH_API std::pair<FaceBitSet, FaceBitSet> findCollidingTriangleBitsets( const MeshPart& a, const MeshPart& b,
    const AffineXf3f* rigidB2A = nullptr );

// finds all pairs of colliding triangles from one mesh or a region
MRMESH_API std::vector<FaceFace> findSelfCollidingTriangles( const MeshPart & mp );
// the same but returns the union of all self-intersecting faces
MRMESH_API FaceBitSet findSelfCollidingTrianglesBS( const MeshPart & mp );

// checks that arbitrary mesh part A is inside of closed mesh part B
MRMESH_API bool isInside( const MeshPart & a, const MeshPart & b, 
    const AffineXf3f * rigidB2A = nullptr ); // rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation

} //namespace MR
