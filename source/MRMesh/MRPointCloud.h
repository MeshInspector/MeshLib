#pragma once

#include "MRVector.h"
#include "MRVector3.h"
#include "MRBitSet.h"
#include "MRMeshFwd.h"
#include "MRSharedThreadSafeOwner.h"
#include "MRCloudPartMapping.h"

namespace MR
{

/// \defgroup PointCloudGroup PointCloud

/// \ingroup PointCloudGroup
struct PointCloud
{
    /// coordinates of points
    VertCoords points;

    /// unit normal directions of points (can be empty if no normals are known)
    VertNormals normals;

    /// only points and normals corresponding to set bits here are valid
    VertBitSet validPoints;

    /// computes the total number of valid points in the cloud
    [[nodiscard]] size_t calcNumValidPoints() const { return validPoints.count(); }

    /// returns true if there is a normal for each point
    [[nodiscard]] bool hasNormals() const { return normals.size() >= points.size(); }

    /// if region pointer is not null then converts it in reference, otherwise returns all valid points in the cloud
    [[nodiscard]] const VertBitSet & getVertIds( const VertBitSet * region ) const
    {
        assert( !region || region->is_subset_of( validPoints ) ); // if region is given, then region must be a subset of valid points
        return region ? *region : validPoints;
    }

    /// returns cached aabb-tree for this point cloud, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePoints& getAABBTree() const;

    /// returns cached aabb-tree for this point cloud, but does not create it if it did not exist
    [[nodiscard]] const AABBTreePoints * getAABBTreeNotCreate() const { return AABBTreeOwner_.get(); }

    /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
    [[nodiscard]] MRMESH_API Box3f getBoundingBox() const;

    /// passes through all valid points and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    /// passes through all given vertices (or all valid vertices if region == null) and finds the minimal bounding box containing all of them
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const VertBitSet * region, const AffineXf3f * toWorld = nullptr ) const;

    /// computes average position of all valid points
    [[nodiscard]] MRMESH_API Vector3f findCenterFromPoints() const;

    /// computes bounding box and returns its center
    [[nodiscard]] MRMESH_API Vector3f findCenterFromBBox() const;

    /// returns all valid point ids sorted lexicographically by their coordinates (optimal for uniform sampling)
    [[nodiscard]] MRMESH_API std::vector<VertId> getLexicographicalOrder() const;

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

    /// flip orientation (normals) of given points (or all valid points is nullptr)
    MRMESH_API void flipOrientation( const VertBitSet * region = nullptr );

    /// tightly packs all arrays eliminating invalid points, but relative order of valid points is preserved;
    /// returns false if the cloud was packed before the call and nothing has been changed;
    /// if pack is done optionally returns mappings: new.id -> old.id
    MRMESH_API bool pack( VertMap * outNew2Old = nullptr );

    /// tightly packs all arrays eliminating invalid points, reorders valid points according to given strategy;
    /// \return points mapping: old -> new
    MRMESH_API VertBMap pack( Reorder reoder );

    /// Invalidates caches (e.g. aabb-tree) after a change in point cloud
    void invalidateCaches() { AABBTreeOwner_.reset(); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    mutable SharedThreadSafeOwner<AABBTreePoints> AABBTreeOwner_;
};

} // namespace MR
