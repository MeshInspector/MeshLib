#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{

constexpr int NoAngleChangeLimit = 10;

struct DeloneSettings
{
    /// Maximal allowed surface deviation during every individual flip
    float maxDeviationAfterFlip = FLT_MAX;
    /// Maximal allowed dihedral angle change over the flipped edge
    float maxAngleChange = NoAngleChangeLimit;
    /// if this value is less than FLT_MAX then the algorithm will try to minimize maximal triangle aspect ratio,
    /// and ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value
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
MRMESH_API bool checkDeloneQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d, double maxAngleChange = NoAngleChangeLimit );
MRMESH_API bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange = NoAngleChangeLimit );

/// given quadrangle ABCD, checks whether its subdivision on 
/// (triangles ABD and DBC) is better than on (triangles ABC and ACD),
/// where "better" means smaller maximum triangle aspect ratio and the flip of edge from BD to AC will
/// not change dihedral angle more than on given threshold
MRMESH_API bool checkAspectRatiosInQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d, double maxAngleChange, double criticalTriAspectRatio );
MRMESH_API bool checkAspectRatiosInQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange, float criticalTriAspectRatio );

/// consider quadrangle formed by left and right triangles of given edge, and
/// checks whether this edge satisfies Delone's condition in the quadrangle;
/// \return false otherwise if flipping the edge does not introduce too large surface deviation (can be returned only for inner edge of the region)
MRMESH_API bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, const DeloneSettings& settings = {} );

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
