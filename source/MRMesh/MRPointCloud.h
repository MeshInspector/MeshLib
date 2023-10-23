#pragma once

#include "MRVector.h"
#include "MRBitSet.h"
#include "MRMeshFwd.h"
#include "MRUniqueThreadSafeOwner.h"
#include "MRCloudPartMapping.h"

namespace MR
{

/// \defgroup PointCloudGroup PointCloud

/// \ingroup PointCloudGroup
struct PointCloud
{
    VertCoords points;
    VertNormals normals;
    /// only points corresponding to set bits here are valid
    VertBitSet validPoints;

    /// returns true if there is a normal for each point
    [[nodiscard]] bool hasNormals() const { return normals.size() >= points.size(); }

    /// returns cached aabb-tree for this point cloud, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePoints& getAABBTree() const;

    /// returns cached aabb-tree for this point cloud, but does not create it if it did not exist
    const AABBTreePoints * getAABBTreeNotCreate() const { return AABBTreeOwner_.get(); }

    /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
    MRMESH_API Box3f getBoundingBox() const;

    /// passes through all valid points and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    /// appends points (and normals if it possible) (from) in addition to this points
    /// if this obj have normals and from obj has not it then don't do anything
    /// \param extNormals if given then they will be copied instead of from.normals
    MRMESH_API void addPartByMask( const PointCloud& from, const VertBitSet& fromVerts, const CloudPartMapping& outMap = {},
        const VertNormals * extNormals = nullptr );

    /// appends a point and returns its VertId
    MRMESH_API VertId addPoint( const Vector3f& point );

    /// appends a point with normal and returns its VertId
    MRMESH_API VertId addPoint( const Vector3f& point, const Vector3f& normal );

    /// reflects the points from a given plane
    MRMESH_API void mirror( const Plane3f& plane );

    /// tightly packs all arrays eliminating invalid points, but relative order of valid points is preserved;
    /// returns false if the cloud was packed before the call and nothing has been changed;
    /// if pack is done optionally returns mappings: new.id -> old.id
    MRMESH_API bool pack( VertMap * outNew2Old = nullptr );

    /// tightly packs all arrays eliminating invalid points, reorders valid points so to put close in space points in close indices;
    /// \return points mapping: old -> new
    MRMESH_API VertBMap packOptimally();

    /// Invalidates caches (e.g. aabb-tree) after a change in point cloud
    void invalidateCaches() { AABBTreeOwner_.reset(); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    mutable UniqueThreadSafeOwner<AABBTreePoints> AABBTreeOwner_;
};

} // namespace MR
