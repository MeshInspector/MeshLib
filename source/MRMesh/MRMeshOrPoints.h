#pragma once

#include "MRMeshPart.h"
#include "MRAffineXf3.h"
#include "MRId.h"
#include <cfloat>
#include <functional>
#include <optional>
#include <variant>

namespace MR
{

/// This class can hold either mesh part or point cloud.
/// It is used for generic algorithms operating with either of them
class MeshOrPoints
{
public:
    MeshOrPoints( const Mesh & mesh ) : var_( MeshPart( mesh ) ) { }
    MeshOrPoints( const MeshPart & mp ) : var_( mp ) { }
    MeshOrPoints( const PointCloud & pc ) : var_( &pc ) { }

    /// if this object holds a mesh part then returns pointer on it, otherwise returns nullptr
    [[nodiscard]] const MeshPart* asMeshPart() const;

    /// if this object holds a point cloud then returns pointer on it, otherwise returns nullptr
    [[nodiscard]] const PointCloud* asPointCloud() const;

    /// returns the minimal bounding box containing all valid vertices of the object (and not only part of mesh);
    /// implemented via obj.getAABBTree()
    [[nodiscard]] MRMESH_API Box3f getObjBoundingBox() const;

    /// if AABBTree is already built does nothing otherwise builds and caches it
    MRMESH_API void cacheAABBTree() const;

    /// passes through all valid vertices and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    /// Adds in existing PointAccumulator the elements of the contained object
    MRMESH_API void accumulate( PointAccumulator& accum, const AffineXf3f* xf = nullptr ) const;

    /// performs sampling of vertices or points;
    /// subdivides bounding box of the object on voxels of approximately given size and returns at most one vertex per voxel;
    /// voxelSize is automatically increased to avoid more voxels than \param maxVoxels;
    /// returns std::nullopt if it was terminated by the callback
    [[nodiscard]] MRMESH_API std::optional<VertBitSet> pointsGridSampling( float voxelSize, size_t maxVoxels = 500000,
        const ProgressCallback & cb = {} ) const;

    /// gives access to points-vector (which can include invalid points as well)
    [[nodiscard]] MRMESH_API const VertCoords & points() const;

    /// gives access to bit set of valid points
    [[nodiscard]] MRMESH_API const VertBitSet& validPoints() const;

    /// returns normals generating function: VertId->normal (or empty for point cloud without normals)
    [[nodiscard]] MRMESH_API std::function<Vector3f(VertId)> normals() const;

    /// returns weights generating function: VertId->float:
    /// for mesh it is double area of surrounding triangles, and for point cloud - nothing
    [[nodiscard]] MRMESH_API std::function<float(VertId)> weights() const;

    struct ProjectionResult
    {
        /// found closest point
        Vector3f point;

        /// normal at the closest point;
        /// for meshes it will be pseudonormal with the differentiation depending on closest point location (face/edge/vertex)
        std::optional<Vector3f> normal;

        /// can be true only for meshes, if the closest point is located on the boundary
        bool isBd = false;

        /// squared distance from query point to the closest point
        float distSq = FLT_MAX;

        /// for point clouds it is the closest vertex,
        /// for meshes it is the closest vertex of the triangle with the closest point
        VertId closestVert;
    };

    /// returns a function that finds projection (closest) points on this: Vector3f->ProjectionResult
    [[nodiscard]] MRMESH_API std::function<ProjectionResult( const Vector3f & )> projector() const;

    using LimitedProjectorFunc = std::function<void( const Vector3f& p, ProjectionResult& res )>;
    /// returns a function that updates projection (closest) points on this,
    /// the update takes place only if res.distSq on input is more than squared distance to the closest point
    [[nodiscard]] MRMESH_API LimitedProjectorFunc limitedProjector() const;

private:
    std::variant<MeshPart, const PointCloud*> var_;
};

/// an object and its transformation to global space with other objects
struct MeshOrPointsXf
{
    MeshOrPoints obj;
    AffineXf3f xf;
};

/// constructs MeshOrPoints from ObjectMesh or ObjectPoints, otherwise returns nullopt
[[nodiscard]] MRMESH_API std::optional<MeshOrPoints> getMeshOrPoints( const VisualObject * obj );

/// to receive object id + projection result on it
using ProjectOnAllCallback = std::function<void( ObjId, MeshOrPoints::ProjectionResult )>;

/// finds closest point on every object within given distance
MRMESH_API void projectOnAll(
    const Vector3f& pt, ///< target point in world coordinates
    const AABBTreeObjects & tree, ///< contains a set of objects to search closest points on each of them
    float upDistLimitSq, ///< upper limit on the distance in question
    const ProjectOnAllCallback & callback, ///< each found closest point within given distance will be returned via this callback
    ObjId skipObjId = {} ); ///< projection on given object will be skipped

inline const MeshPart* MeshOrPoints::asMeshPart() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) { return &mp; },
        []( const PointCloud * ) { return (const MeshPart*)nullptr; }
    }, var_ );
}

inline const PointCloud* MeshOrPoints::asPointCloud() const
{
    return std::visit( overloaded{
        []( const MeshPart & ) { return (const PointCloud *)nullptr; },
        []( const PointCloud * pc ) { return pc; }
    }, var_ );
}

} // namespace MR
