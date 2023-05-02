#pragma once
#include "exports.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRMesh.h"
namespace MR
{
namespace Cuda
{
struct FastWindingNumberData;
/// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
/// \ingroup AABBTreeGroup
class FastWindingNumber : public IFastWindingNumber
{
    std::shared_ptr<FastWindingNumberData> data_;

public:
    /// constructs this from AABB tree of given mesh;
    MRCUDA_API FastWindingNumber( const Mesh& mesh );
    /// <summary>
    /// calculates winding numbers for a vector of points
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="points">incoming points</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    /// <param name="skipFace">this triangle (if it is close to \param q) will be skipped from summation</param>
    MRCUDA_API void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) override;
    /// <summary>
    /// calculates winding numbers for all centers of mesh's triangles. if winding number is less than 0 or greater then 1, that face is marked as self-intersected
    /// </summary>
    /// <param name="res">resulting bit set</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    MRCUDA_API void calcSelfIntersections( FaceBitSet& res, float beta ) override;
    /// <summary>
    /// calculates winding numbers for each point in a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="minCoord">minimal coordinates of grid points</param>
    /// <param name="voxelSize">size of voxel</param>
    /// <param name="gridToMeshXf">transform from grid to mesh</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    MRCUDA_API void calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta ) override;
    /// <summary>
    /// calculates winding numbers for each point in a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="minCoord">minimal coordinates of grid points</param>
    /// <param name="voxelSize">size of voxel</param>
    /// <param name="gridToMeshXf">transform from grid to mesh</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    MRCUDA_API void calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, float maxDistSq = FLT_MAX, float minDistSq = 0 ) override;
};

}
}
