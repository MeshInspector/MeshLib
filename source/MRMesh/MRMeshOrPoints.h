#pragma once

#include "MRMeshPart.h"
#include "MRVector3.h"
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

    /// passes through all valid vertices and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    /// Adds in existing PointAccumulator the elements of the contained object
    MRMESH_API void accumulate( PointAccumulator& accum, const AffineXf3f* xf = nullptr ) const;

    /// performs sampling of vertices or points;
    /// subdivides bounding box of the object on voxels of approximately given size and returns at most one vertex per voxel;
    /// returns std::nullopt if it was terminated by the callback
    [[nodiscard]] MRMESH_API std::optional<VertBitSet> pointsGridSampling( float voxelSize, const ProgressCallback & cb = {} );

    /// gives access to points-vector (which can include invalid points as well)
    [[nodiscard]] MRMESH_API const VertCoords & points() const;

    /// returns normals generating function: VertId->normal (or empty for point cloud without normals)
    [[nodiscard]] MRMESH_API std::function<Vector3f(VertId)> normals() const;

    /// returns weights generating function: VertId->float:
    /// for mesh it is double area of surrounding triangles, and for point cloud - nothing
    [[nodiscard]] MRMESH_API std::function<float(VertId)> weights() const;

    struct ProjectionResult
    {
        /// found projection point
        Vector3f point;
        /// normal at projection point
        std::optional<Vector3f> normal;
        /// can be true only for meshes, if projection point is located on the boundary
        bool isBd = false;
        /// squared distance from query point to projection point
        float distSq = 0;
    };

    /// returns a function that finds projection (closest) points on this: Vector3f->ProjectionResult
    [[nodiscard]] MRMESH_API std::function<ProjectionResult( const Vector3f & )> projector() const;

private:
    std::variant<MeshPart, const PointCloud*> var_;
};

/// constructs MeshOrPoints from ObjectMesh or ObjectPoints, otherwise returns nullopt
[[nodiscard]] MRMESH_API std::optional<MeshOrPoints> getMeshOrPoints( const VisualObject * obj );

} // namespace MR
