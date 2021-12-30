#pragma once
#include "exports.h"
#include "MREBooleanOperation.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"

namespace MRE
{

/** \defgroup BooleanGroup Surface Boolean overview
  * \brief Chapter about Constructive Solid Geometry operations
  * 
  * This chapter represents documentation of MeshRusExt CSG
  */


/** \struct MRE::BooleanResult
  * \ingroup BooleanGroup
  * \brief Structure contain boolean result
  * 
  * This structure store result mesh of MRE::boolean or some error info
  */
struct BooleanResult
{    
    /// Result mesh of boolean operation, if error occurred it would be empty
    MR::Mesh mesh;
    /// If input contours have intersections, this face bit set presents faces of mesh `A` on which contours intersect
    MR::FaceBitSet meshABadContourFaces;
    /// If input contours have intersections, this face bit set presents faces of mesh `B` on which contours intersect
    MR::FaceBitSet meshBBadContourFaces;
    /// Holds error message, empty if boolean succeed
    std::string errorString;
    /// Returns true if boolean succeed, false otherwise
    bool valid() const { return errorString.empty(); }
    MR::Mesh& operator*() { return mesh; }
    const MR::Mesh& operator*() const { return mesh; }
    MR::Mesh* operator->() { return &mesh; }
    const MR::Mesh* operator->() const { return &mesh; }
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
MREALGORITHMS_API BooleanResult boolean( const MR::Mesh& meshA, const MR::Mesh& meshB, BooleanOperation operation,
                                         const MR::AffineXf3f* rigidB2A = nullptr, BooleanResultMapper* mapper = nullptr );

MREALGORITHMS_API void loadMREAlgorithmsDll();

}