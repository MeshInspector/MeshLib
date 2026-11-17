#pragma once

#include "MRPointOnFace.h"
#include "MRTriPoint.h"
#include "MRLine3.h"
#include "MRMeshPart.h"
#include "MRMeshTriPoint.h"
#include <cfloat>
#include <functional>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct MeshIntersectionResult
{
    /// stores intersected face and global coordinates
    PointOnFace proj;

    /// barycentric coordinates of proj relative to the vertices of proj.face in the order of MeshTopology::getTriVerts( proj.face )
    TriPointf tp;

    /// stores the distance from ray origin to the intersection point in direction units
    float distanceAlongLine = 0;

    /// check for validity
    [[nodiscard]] bool valid() const { return proj.face.valid(); }
    [[nodiscard]] explicit operator bool() const { return proj.face.valid(); }

    /// returns the intersection point as MeshTriPoint relative to topology.edgeWithLeft( proj.face ), or invalid MeshTriPoint if no intersection was found
    [[nodiscard]] MRMESH_API MeshTriPoint mtp( const MeshTopology & topology ) const;
};

/// Finds ray and mesh intersection in float-precision.
/// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
/// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
/// \p vadidFaces if given then all faces for which false is returned will be skipped
/// Finds the closest intersection to ray origin (line param=0)
/// or any intersection for better performance if \p !closestIntersect.
[[nodiscard]] MRMESH_API MeshIntersectionResult rayMeshIntersect( const MeshPart& meshPart, const Line3f& line,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr, bool closestIntersect = true,
    const FacePredicate & validFaces = {} );

/// Finds ray and mesh intersection in double-precision.
/// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
/// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
/// \p vadidFaces if given then all faces for which false is returned will be skipped
/// Finds the closest intersection to ray origin (line param=0)
/// or any intersection for better performance if \p !closestIntersect.
[[nodiscard]] MRMESH_API MeshIntersectionResult rayMeshIntersect( const MeshPart& meshPart, const Line3d& line,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr, bool closestIntersect = true,
    const FacePredicate & validFaces = {} );

/// Same as rayMeshIntersect above, but for the triangles of given tree (only from \p region if given), e.g. AABBTree( tmesh ),
/// with \p points the coordinates of their vertices; tp of the result is relative to the vertices of the triangle in the tree
[[nodiscard]] MRMESH_API MeshIntersectionResult rayMeshIntersect( const VertCoords & points, const AABBTree & tree, const Line3f& line,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr, bool closestIntersect = true,
    const FacePredicate & validFaces = {}, const FaceBitSet * region = nullptr );

/// Same as rayMeshIntersect above in double-precision, but for the triangles of given tree (only from \p region if given), e.g. AABBTree( tmesh ),
/// with \p points the coordinates of their vertices; tp of the result is relative to the vertices of the triangle in the tree
[[nodiscard]] MRMESH_API MeshIntersectionResult rayMeshIntersect( const VertCoords & points, const AABBTree & tree, const Line3d& line,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr, bool closestIntersect = true,
    const FacePredicate & validFaces = {}, const FaceBitSet * region = nullptr );

struct MultiRayMeshIntersectResult
{
    // outputs (each one if optional) for every ray:
    BitSet * intersectingRays = nullptr;         ///< true if the ray has intersection with mesh part, false otherwise
    std::vector<float> * rayDistances = nullptr;    ///< distance along each ray till the intersection point or NaN if no intersection
    std::vector<FaceId> * isectFaces = nullptr;  ///< intersected triangles from mesh
    std::vector<TriPointf> * isectBary = nullptr;///< barycentric coordinates of the intersection point within intersected triangle or NaNs if no intersection
    std::vector<Vector3f> * isectPts = nullptr;  ///< intersection points or NaNs if no intersection
};

/// Finds intersections between a mesh and multiple rays in parallel (in float-precision).
/// \p rayStart and \p rayEnd define the interval on all rays to detect an intersection.
/// \p vadidFaces if given then all faces for which false is returned will be skipped
MRMESH_API void multiRayMeshIntersect(
    // input:
    const MeshPart& meshPart, ///< mesh (or its part) to find intersections with
    const std::vector<Vector3f>& origins, ///< origin point of every ray
    const std::vector<Vector3f>& dirs,    ///< direction of every ray
    const MultiRayMeshIntersectResult& result, ///< output data for every ray
    // advanced options:
    float rayStart = 0.0f, float rayEnd = FLT_MAX,
    bool closestIntersect = true, ///< whether to search for the closest intersection to ray origin (line param=0), or any intersection
    const FacePredicate & validFaces = {} ///< if given then all faces for which false is returned will be skipped
);

/// Same as multiRayMeshIntersect above, but for the triangles of given tree (only from \p region if given), e.g. AABBTree( tmesh ),
/// with \p points the coordinates of their vertices
MRMESH_API void multiRayMeshIntersect(
    const VertCoords & points, const AABBTree & tree,
    const std::vector<Vector3f>& origins,
    const std::vector<Vector3f>& dirs,
    const MultiRayMeshIntersectResult& result,
    float rayStart = 0.0f, float rayEnd = FLT_MAX,
    bool closestIntersect = true,
    const FacePredicate & validFaces = {},
    const FaceBitSet * region = nullptr
);

struct MultiMeshIntersectionResult : MeshIntersectionResult
{
    /// the intersection found in this mesh
    const Mesh * mesh = nullptr;
};

template<typename T>
struct Line3Mesh
{
    Line3<T> line; ///< in the reference frame of mesh
    IntersectionPrecomputes<T> * prec = nullptr; ///< set it to a valid pointer for better performance
    const Mesh * mesh = nullptr; ///< must be set a valid pointer before use
    const AABBTree * tree = nullptr; ///< not used: the rays take mesh->getAABBTree()
    const FaceBitSet * region = nullptr; ///< may remain nullptr, meaning all mesh
};
using Line3fMesh = Line3Mesh<float>;
using Line3dMesh = Line3Mesh<double>;

/// Intersects ray with many meshes. Finds any intersection (not the closest)
/// \anchor rayMultiMeshAnyIntersectF
[[nodiscard]] MRMESH_API MultiMeshIntersectionResult rayMultiMeshAnyIntersect( const std::vector<Line3fMesh> & lineMeshes,
    float rayStart = 0.0f, float rayEnd = FLT_MAX );
/// Same as \ref rayMultiMeshAnyIntersectF, but use double precision
[[nodiscard]] MRMESH_API MultiMeshIntersectionResult rayMultiMeshAnyIntersect( const std::vector<Line3dMesh> & lineMeshes,
    double rayStart = 0.0, double rayEnd = DBL_MAX );

/// this callback is envoked for each encountered ray-mesh intersection;
/// if it returns false, then the search immediately terminates
using MeshIntersectionCallback = std::function<bool(const MeshIntersectionResult &)>;
/// Intersects ray with mesh. Finds all intersections
/// \anchor rayMeshIntersectAllF
MRMESH_API void rayMeshIntersectAll( const MeshPart& meshPart, const Line3f& line, MeshIntersectionCallback callback,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr );
/// Same as \ref rayMeshIntersectAllF, but use double precision
MRMESH_API void rayMeshIntersectAll( const MeshPart& meshPart, const Line3d& line, MeshIntersectionCallback callback,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr );

/// Same as \ref rayMeshIntersectAllF, but for the triangles of given tree (only from \p region if given), e.g. AABBTree( tmesh ),
/// with \p points the coordinates of their vertices
MRMESH_API void rayMeshIntersectAll( const VertCoords & points, const AABBTree & tree, const Line3f& line, MeshIntersectionCallback callback,
    float rayStart = 0.0f, float rayEnd = FLT_MAX, const IntersectionPrecomputes<float>* prec = nullptr, const FaceBitSet * region = nullptr );
/// Same as \ref rayMeshIntersectAllF in double precision, but for the triangles of given tree (only from \p region if given), e.g. AABBTree( tmesh ),
/// with \p points the coordinates of their vertices
MRMESH_API void rayMeshIntersectAll( const VertCoords & points, const AABBTree & tree, const Line3d& line, MeshIntersectionCallback callback,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr, const FaceBitSet * region = nullptr );

/// this callback is envoked for each encountered ray-mesh intersection,
/// fromFront is true if the ray crosses the triangle from its front side (where the triangle's normal points) to its back,
/// i.e. enters a closed mesh with outward normals;
/// if it returns false, then the search immediately terminates
using MeshIntersectionWithSideCallback = std::function<bool( const MeshIntersectionResult &, bool fromFront )>;
/// Same as \ref rayMeshIntersectAllF in double precision, also reporting the side from which the ray crosses each triangle;
/// both the intersections and the sides are decided by precise predicates, so for a closed mesh
/// every infinite ray has the same number of crossings from front and from back
MRMESH_API void rayMeshIntersectAll( const MeshPart& meshPart, const Line3d& line, MeshIntersectionWithSideCallback callback,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr );
/// Same as above, but for the triangles of given tree (only from \p region if given), e.g. AABBTree( tmesh ),
/// with \p points the coordinates of their vertices
MRMESH_API void rayMeshIntersectAll( const VertCoords & points, const AABBTree & tree, const Line3d& line, MeshIntersectionWithSideCallback callback,
    double rayStart = 0.0, double rayEnd = DBL_MAX, const IntersectionPrecomputes<double>* prec = nullptr, const FaceBitSet * region = nullptr );

/// given mesh part and arbitrary plane, outputs
/// \param fs  triangles from boxes crossed or touched by the plane
/// \param ues edges of these triangles
/// \param vs  vertices of these triangles
/// \param fsVec triangles from boxes crossed or touched by the plane in unspecified order
MRMESH_API void planeMeshIntersect( const MeshPart& meshPart, const Plane3f & plane,
    FaceBitSet * fs, UndirectedEdgeBitSet * ues, VertBitSet * vs,
    std::vector<FaceId> * fsVec = nullptr );

/// given the tree of triangles (only from \p region if given), e.g. AABBTree( tmesh ), and arbitrary plane, outputs
/// \param fs  triangles from boxes crossed or touched by the plane (resized if necessary)
/// \param vs  vertices of these triangles (resized if necessary)
/// \param fsVec triangles from boxes crossed or touched by the plane in unspecified order
MRMESH_API void planeMeshIntersect( const AABBTree & tree, const Plane3f & plane,
    FaceBitSet * fs, VertBitSet * vs, std::vector<FaceId> * fsVec = nullptr, const FaceBitSet * region = nullptr );

/// given mesh part and plane z=zLevel, outputs
/// \param fs  triangles crossed or touched by the plane
/// \param ues edges of these triangles
/// \param vs  vertices of these triangles
/// \param fsVec triangles crossed or touched by the plane in unspecified order
MRMESH_API void xyPlaneMeshIntersect( const MeshPart& meshPart, float zLevel,
    FaceBitSet * fs, UndirectedEdgeBitSet * ues, VertBitSet * vs,
    std::vector<FaceId> * fsVec = nullptr );

/// given the tree of triangles (only from \p region if given), e.g. AABBTree( tmesh ), and plane z=zLevel, outputs
/// \param fs  triangles crossed or touched by the plane (resized if necessary)
/// \param vs  vertices of these triangles (resized if necessary)
/// \param fsVec triangles crossed or touched by the plane in unspecified order
MRMESH_API void xyPlaneMeshIntersect( const AABBTree & tree, float zLevel,
    FaceBitSet * fs, VertBitSet * vs, std::vector<FaceId> * fsVec = nullptr, const FaceBitSet * region = nullptr );

/// \}

} // namespace MR
