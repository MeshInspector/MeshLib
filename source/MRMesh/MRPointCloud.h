#pragma once

#include "MRVector.h"
#include "MRBitSet.h"
#include "MRMeshFwd.h"
#include "MRUniqueThreadSafeOwner.h"

namespace MR
{

struct PointCloud
{
public:
    VertCoords points;
    Vector<Vector3f, VertId> normals;

    VertBitSet validPoints;

    // returns cached aabb-tree for this point cloud, creating it if it did not exist in a thread-safe manner
    const AABBTreePoints& getAABBTree() const;
    // returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
    MRMESH_API Box3f getBoundingBox() const;
    // passes through all valid points and finds the minimal bounding box containing all of them;
    // if toWorld transformation is given then returns minimal bounding box in world space
    MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    // Invalidates caches (e.g. aabb-tree) after a change in point cloud
    void invalidateCaches() { AABBTreeOwner_.reset(); }

private:
    mutable UniqueThreadSafeOwner<AABBTreePoints> AABBTreeOwner_;
};

} //namespace MR
