#pragma once

#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRId.h"

namespace MR
{

/// Abstract class for fast approximate computation of generalized winding number for a mesh (using its AABB tree)
class IFastWindingNumber
{
public:
    virtual ~IFastWindingNumber() = default;

    /// <summary>
    /// calculates winding numbers in the points from given vector
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="points">incoming points</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    /// <param name="skipFace">this triangle (if it is close to `q`) will be skipped from summation</param>
    virtual void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) = 0;

    /// <summary>
    /// calculates winding numbers for all centers of mesh's triangles. if winding number is less than 0 or greater then 1, that face is marked as self-intersected
    /// </summary>
    /// <param name="res">resulting bit set</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    /// <returns>false if the operation was canceled by the user</returns>
    virtual bool calcSelfIntersections( FaceBitSet& res, float beta, ProgressCallback cb = {} ) = 0;

    /// <summary>
    /// calculates winding numbers in each point from a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="gridToMeshXf">transform from integer grid locations to voxel's centers in mesh reference frame</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    virtual Expected<void> calcFromGrid( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb = {} ) = 0;

    /// <summary>
    /// calculates distances with the sign obtained from winding number in each point from a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting signed distances, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="gridToMeshXf">transform from integer grid locations to voxel's centers in mesh reference frame</param>
    /// <param name="windingNumberThreshold">positive distance if winding number below or equal this threshold</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    virtual Expected<void> calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float windingNumberThreshold, float beta, float maxDistSq, float minDistSq, ProgressCallback cb ) = 0;
};

/// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
/// \ingroup AABBTreeGroup
/// Note, this used to be `[[nodiscard]]`, but GCC 12 doesn't understand both `[[...]]` and `__attribute__(...)` on the same class.
/// A possible fix is to change `MRMESH_CLASS` globally to `[[__gnu__::__visibility__("default")]]`.
class MRMESH_CLASS FastWindingNumber : public IFastWindingNumber
{
public:
    /// constructs this from AABB tree of given mesh;
    /// this remains valid only if tree is valid
    [[nodiscard]] MRMESH_API FastWindingNumber( const Mesh & mesh );

    // see methods' descriptions in IFastWindingNumber
    MRMESH_API void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) override;
    MRMESH_API bool calcSelfIntersections( FaceBitSet& res, float beta, ProgressCallback cb ) override;
    MRMESH_API Expected<void> calcFromGrid( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb ) override;
    MRMESH_API float calcWithDistances( const Vector3f& p, float windingNumberThreshold, float beta, float maxDistSq, float minDistSq );
    MRMESH_API Expected<void> calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float windingNumberThreshold, float beta, float maxDistSq, float minDistSq, ProgressCallback cb ) override;

private:
    [[nodiscard]] float calc_( const Vector3f & q, float beta, FaceId skipFace = {} ) const;

    const Mesh & mesh_;
    const AABBTree & tree_;
    const Dipoles & dipoles_;
};

} // namespace MR
