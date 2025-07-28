#pragma once

#include "MRBooleanOperation.h"
#include "MRContoursCut.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRExpected.h"
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
  * \snippet cpp-examples/MeshBoolean.dox.cpp 0
  *
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
                                  const AffineXf3f* rigidB2A, BooleanResultMapper* mapper = nullptr, ProgressCallback cb = {} );
MRMESH_API BooleanResult boolean( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation,
                                  const AffineXf3f* rigidB2A, BooleanResultMapper* mapper = nullptr, ProgressCallback cb = {} );


struct BooleanPreCutResult
{
    Mesh mesh;
    OneMeshContours contours;
};

/** \struct MR::BooleanResult
  * \ingroup BooleanGroup
  * \brief Structure with parameters for boolean call
  */
struct BooleanParameters
{
    /// Transform from mesh `B` space to mesh `A` space
    const AffineXf3f* rigidB2A = nullptr;
    
    /// Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology
    BooleanResultMapper* mapper = nullptr;
    
    /// Optional precut output of meshA, if present - does not perform boolean and just return them
    BooleanPreCutResult* outPreCutA = nullptr;
    
    /// Optional precut output of meshB, if present - does not perform boolean and just return them
    BooleanPreCutResult* outPreCutB = nullptr;
    
    /// Optional output cut edges of booleaned meshes 
    std::vector<EdgeLoop>* outCutEdges = nullptr;
    
    /// By default produce valid operation on disconnected components
    /// if set merge all non-intersecting components
    bool mergeAllNonIntersectingComponents = false;
    
    /// If this option is enabled boolean will try to cut meshes even if there are self-intersections in intersecting area
    /// it might work in some cases, but in general it might prevent fast error report and lead to other errors along the way
    /// \warning not recommended in most cases
    bool forceCut = false;
    
    ProgressCallback cb = {};
};

MRMESH_API BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation,
                                  const BooleanParameters& params = {} );
MRMESH_API BooleanResult boolean( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation,
                                  const BooleanParameters& params = {} );

/// performs boolean operation on mesh with itself, cutting simple intersections contours and flipping their connectivity
/// this function is experimental and likely to change signature and/or behavior in future 
MRMESH_API Expected<Mesh> selfBoolean( const Mesh& mesh );

/// returns intersection contours of given meshes
MRMESH_API Contours3f findIntersectionContours( const Mesh& meshA, const Mesh& meshB, const AffineXf3f* rigidB2A = nullptr );

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
 MRMESH_API Expected<BooleanResultPoints,std::string> getBooleanPoints( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation,
                                                               const AffineXf3f* rigidB2A = nullptr );


/// converts all vertices of the mesh first in integer-coordinates, and then back in float-coordinates;
/// this is necessary to avoid small self-intersections near newly introduced vertices on cut-contours;
/// this actually sligntly moves only small perentage of vertices near x=0 or y=0 or z=0 planes, where float-precision is higher than int-precision;
/// newly introduced vertices on cut-contours are also converted, but we expected that they remain unchanged due to idempotent property of the conversion
MRMESH_API void convertIntFloatAllVerts( Mesh & mesh, const CoordinateConverters& conv );

} //namespace MR
