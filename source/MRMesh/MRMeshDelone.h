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

    /// Only edges with left and right faces in this set can be flipped
    const FaceBitSet* region = nullptr;

    /// Edges specified by this bit-set will never be flipped
    const UndirectedEdgeBitSet* notFlippable = nullptr;

    /// Only edges with origin or destination in this set before or after flip can be flipped
    const VertBitSet* vertRegion = nullptr;
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
[[nodiscard]] MRMESH_API bool checkDeloneQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d, double maxAngleChange = DBL_MAX );
/// converts arguments in double and calls above function
[[nodiscard]] MRMESH_API bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float maxAngleChange = FLT_MAX );

enum class FlipEdge : int
{
    Can,    ///< edge flipping is possible
    Cannot, ///< edge flipping is prohibited by topology or by constraints
    Must    ///< edge flipping is required to solve some topology issue
};

/// consider topology and constraints to decide about flip possibility
[[nodiscard]] MRMESH_API FlipEdge canFlipEdge( const MeshTopology & topology, EdgeId edge,
    const FaceBitSet* region = nullptr,
    const UndirectedEdgeBitSet* notFlippable = nullptr,
    const VertBitSet* vertRegion = nullptr );

/// consider quadrangle formed by left and right triangles of given edge, and
/// checks whether this edge satisfies Delone's condition in the quadrangle;
/// \return false otherwise if flipping the edge does not introduce too large surface deviation (can be returned only for inner edge of the region)
[[nodiscard]] MRMESH_API bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, const DeloneSettings& settings = {},
    float * deviationSqAfterFlip = nullptr ); ///< squared surface deviation after flip is written here (at least when the function returns false)
[[nodiscard]] MRMESH_API bool checkDeloneQuadrangleInMesh( const MeshTopology & topology, const VertCoords & points, EdgeId edge, const DeloneSettings& settings = {},
    float * deviationSqAfterFlip = nullptr ); ///< squared surface deviation after flip is written here (at least when the function returns false)

/// given quadrangle ABCD, selects how to best triangulate it:
///   false = by introducing BD diagonal and splitting ABCD on triangles ABD and DBC,
///   true  = by introducing AC diagonal and splitting ABCD on triangles ABC and ACD
[[nodiscard]] MRMESH_API bool bestQuadrangleDiagonal( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d );

/// improves mesh triangulation in a ring of vertices with common origin and represented by edge e
MRMESH_API void makeDeloneOriginRing( Mesh & mesh, EdgeId e, const DeloneSettings& settings = {} );
MRMESH_API void makeDeloneOriginRing( MeshTopology& topology, const VertCoords& points, EdgeId e, const DeloneSettings& settings = {} );

/// improves mesh triangulation by performing flipping of edges to satisfy Delone local property,
/// consider every edge at most numIters times, and allow surface deviation at most on given value during every individual flip,
/// \return the number of flips done
/// \param numIters Maximal iteration count
/// \param progressCallback Callback to report algorithm progress and cancel it by user request
MRMESH_API int makeDeloneEdgeFlips( Mesh & mesh, const DeloneSettings& settings = {}, int numIters = 1, const ProgressCallback& progressCallback = {} );
MRMESH_API int makeDeloneEdgeFlips( MeshTopology& topology, const VertCoords& points, const DeloneSettings& settings = {}, int numIters = 1, const ProgressCallback& progressCallback = {} );

struct IntrinsicDeloneSettings
{
    /// the edge is considered Delaunay, if cotan(a1) + cotan(a2) >= threshold;
    /// passing positive(negative) threshold makes less(more) edges satisfy Delaunay conditions
    float threshold = 0;

    /// Only edges with left and right faces in this set can be flipped
    const FaceBitSet* region = nullptr;

    /// Edges specified by this bit-set will never be flipped
    const UndirectedEdgeBitSet* notFlippable = nullptr;

    /// Only edges with origin or destination in this set before or after flip can be flipped
    const VertBitSet* vertRegion = nullptr;
};

/// improves mesh triangulation by performing flipping of edges to satisfy Intrinsic Delaunay local property,
/// consider every edge at most numIters times,
/// \return the number of flips done
/// \param numIters Maximal iteration count
/// \param progressCallback Callback to report algorithm progress and cancel it by user request
/// see "An Algorithm for the Construction of Intrinsic Delaunay Triangulations with Applications to Digital Geometry Processing". https://page.math.tu-berlin.de/~bobenko/papers/InDel.pdf
MRMESH_API int makeDeloneEdgeFlips( EdgeLengthMesh & mesh, const IntrinsicDeloneSettings& settings = {}, int numIters = 1, const ProgressCallback& progressCallback = {} );

/// \}

} // namespace MR
