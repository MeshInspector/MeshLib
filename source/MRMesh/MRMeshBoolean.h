#pragma once

#include "MRBooleanOperation.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include <tl/expected.hpp>
#include <string>

namespace MR
{

/** \defgroup BooleanGroup Surface Boolean overview
  * \brief Chapter about Constructive Solid Geometry operations
  * 
  * This chapter represents documentation of MeshLib CSG
  */


/** \struct MR::BooleanResult
  * \ingroup BooleanGroup
  * \brief Structure contain boolean result
  * 
  * This structure store result mesh of MR::boolean or some error info
  */
struct BooleanResult
{    
    /// Result mesh of boolean operation, if error occurred it would be empty
    Mesh mesh;
    /// If input contours have intersections, this face bit set presents faces of mesh `A` on which contours intersect
    FaceBitSet meshABadContourFaces;
    /// If input contours have intersections, this face bit set presents faces of mesh `B` on which contours intersect
    FaceBitSet meshBBadContourFaces;
    /// Holds error message, empty if boolean succeed
    std::string errorString;
    /// Returns true if boolean succeed, false otherwise
    bool valid() const { return errorString.empty(); }
    Mesh& operator*() { return mesh; }
    const Mesh& operator*() const { return mesh; }
    Mesh* operator->() { return &mesh; }
    const Mesh* operator->() const { return &mesh; }
    operator bool()const { return valid(); }
};

/** \brief Performs CSG operation on two meshes
* 
  * \ingroup BooleanGroup
  * Makes new mesh - result of boolean operation on mesh `A` and mesh `B`
  * \param meshA Input mesh `A`
  * \param meshB Input mesh `B`
  * \param operation CSG operation to perform
  * \param rigidB2A Transform from mesh `B` space to mesh `A` space
  * \param mapper Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology
  * 
  * \note Input meshes should have no self-intersections in intersecting zone
  * \note If meshes are not closed in intersecting zone some boolean operations are not allowed (as far as input meshes interior and exterior cannot be determined)
  */
MRMESH_API BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation,
                                  const AffineXf3f* rigidB2A = nullptr, BooleanResultMapper* mapper = nullptr );
MRMESH_API BooleanResult boolean( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation,
                                  const AffineXf3f* rigidB2A = nullptr, BooleanResultMapper* mapper = nullptr );

/// vertices and points representing mesh intersection result
struct BooleanResultPoints
{
    VertBitSet meshAVerts;
    VertBitSet meshBVerts;
    std::vector<Vector3f> intersectionPoints;
};

/** \brief Returns the points of mesh boolean's result mesh
 *
 * \ingroup BooleanGroup
 * Returns vertices and intersection points of mesh that is result of boolean operation of mesh `A` and mesh `B`.
 * Can be used as fast alternative for cases where the mesh topology can be ignored (bounding box, convex hull, etc.)
 * \param meshA Input mesh `A`
 * \param meshB Input mesh `B`
 * \param operation Boolean operation to perform
 * \param rigidB2A Transform from mesh `B` space to mesh `A` space
 */
 MRMESH_API tl::expected<BooleanResultPoints,std::string> getBooleanPoints( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation,
                                                               const AffineXf3f* rigidB2A = nullptr );

} //namespace MR
