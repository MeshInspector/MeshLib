#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{

struct DeloneSettings
{
    /// Maximal allowed surface deviation during every individual flip
    float maxDeviationAfterFlip = FLT_MAX;
    /// Maximal allowed dihedral angle change (in radians) over the flipped edge
    float maxAngleChange = FLT_MAX;
    /// if this value is less than FLT_MAX then the algorithm will
    /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
    float criticalTriAspectRatio = FLT_MAX;
    /// Region on mesh to be processed, it is constant and not updated
    const FaceBitSet* region = nullptr;
    /// Edges specified by this bit-set will never be flipped
    const UndirectedEdgeBitSet* notFlippable = nullptr;
};

/// \defgroup MeshDeloneGroup Mesh Delone
/// \details https:///en.wikipedia.org/wiki/Boris_Delaunay
/// \ingroup MeshAlgorithmGroup
/// \{

/// given quadrangle ABCD, checks whether its edge AC satisfies Delone's condition;
/// if dihedral angles
///   1) between triangles ABD and DBC and
///   2) between triangles ABC and ACD
/// differ more than on maxAngleChange then also returns true to prevent flipping from 1) to 2)
MRMESH_API bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange = FLT_MAX );

/// consider quadrangle formed by left and right triangles of given edge, and
/// checks whether this edge satisfies Delone's condition in the quadrangle;
/// \return false otherwise if flipping the edge does not introduce too large surface deviation (can be returned only for inner edge of the region)
MRMESH_API bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, const DeloneSettings& settings = {},
    float * deviationSqAfterFlip = nullptr ); ///< squared surface deviation after flip is written here (at least when the function returns false)

/// improves mesh triangulation by performing flipping of edges to satisfy Delone local property,
/// consider every edge at most numIters times, and allow surface deviation at most on given value during every individual flip,
/// \return the number of flips done
/// \param numIters Maximal iteration count
/// \param progressCallback Callback to report algorithm progress and cancel it by user request
MRMESH_API int makeDeloneEdgeFlips( Mesh & mesh, const DeloneSettings& settings = {}, int numIters = 1, ProgressCallback progressCallback = {} );

/// improves mesh triangulation in a ring of vertices with common origin and represented by edge e
MRMESH_API void makeDeloneOriginRing( Mesh & mesh, EdgeId e, const DeloneSettings& settings = {} );

/// \}

} // namespace MR
