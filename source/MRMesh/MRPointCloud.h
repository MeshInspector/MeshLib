#pragma once

#include "MRVector.h"
#include "MRBitSet.h"
#include "MRMeshFwd.h"
#include "MRUniqueThreadSafeOwner.h"

namespace MR
{

/// \defgroup PointCloudGroup PointCloud

/// \ingroup PointCloudGroup
struct PointCloud
{
public:
    VertCoords points;
    Vector<Vector3f, VertId> normals;

    VertBitSet validPoints;

    /// returns cached aabb-tree for this point cloud, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePoints& getAABBTree() const;
    /// returns cached aabb-tree for this point cloud, but does not create it if it did not exist
    const AABBTreePoints * getAABBTreeNotCreate() const { return AABBTreeOwner_.get(); }

    /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
    MRMESH_API Box3f getBoundingBox() const;
    /// passes through all valid points and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    //// appends points (and normals if it possible) (from) in addition to this points
    //// if this obj have normals and from obj has not it then don't do anything
    MRMESH_API void addPartByMask( const PointCloud& from, const VertBitSet& fromVerts, VertMap* oldToNewMap = nullptr );

    /// appends a point and returns its VertId
    MRMESH_API VertId addPoint( const Vector3f& point );
    /// appends a point with normal and returns its VertId
    MRMESH_API VertId addPoint( const Vector3f& point, const Vector3f& normal );

    /// Invalidates caches (e.g. aabb-tree) after a change in point cloud
    void invalidateCaches() { AABBTreeOwner_.reset(); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    mutable UniqueThreadSafeOwner<AABBTreePoints> AABBTreeOwner_;
};

} // namespace MR
