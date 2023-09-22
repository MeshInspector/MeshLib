#pragma once

#include "MRMeshPart.h"
#include <variant>

namespace MR
{

/// This class can hold either mesh part or point cloud.
/// It is used for generic algorithms operating with either of them
class MeshOrPoints
{
public:
    MeshOrPoints( const MeshPart & mp ) : var_( mp ) { }
    MeshOrPoints( const PointCloud & pc ) : var_( &pc ) { }

    /// passes through all valid vertices and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    /// performs sampling of vertices or points;
    /// subdivides bounding box of the object on voxels of approximately given size and returns at most one vertex per voxel;
    /// returns std::nullopt if it was terminated by the callback
    [[nodiscard]] MRMESH_API std::optional<VertBitSet> pointsGridSampling( float voxelSize, const ProgressCallback & cb = {} );

    /// gives access to points-vector (which can include invalid points as well)
    [[nodiscard]] MRMESH_API const VertCoords & points() const;

private:
    std::variant<MeshPart, const PointCloud*> var_;
};

} // namespace MR
