#pragma once

#include "MRAABBTree.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <array>
#include <string>

namespace MR
{

struct Dipole
{
    Vector3f areaPos;
    float area = 0;
    Vector3f dirArea;
    float rr = 0; // maximum squared distance from pos to any corner of the bounding box
    [[nodiscard]] Vector3f pos() const
    {
        return area > 0 ? areaPos / area : areaPos;
    }
    /// returns true if this dipole is good approximation for a point \param q
    [[nodiscard]] bool goodApprox( const Vector3f& q, float beta ) const
    {
        return ( q - pos() ).lengthSq() > sqr( beta ) * rr;
    }
    /// contribution of this dipole to the winding number at point \param q
    [[nodiscard]] float w( const Vector3f& q ) const;
};

static_assert( sizeof( Dipole ) == 8 * sizeof( float ) );
using Dipoles = Vector<Dipole, AABBTree::NodeId>;

/// <summary>
/// calculates dipoles for given mesh and AABB-tree
/// </summary>
MRMESH_API void calcDipoles( Dipoles& dipoles, const AABBTree& tree, const Mesh& mesh );

/// Abstract class for fast approximate computation of winding number for a mesh (using its AABB tree)
class IFastWindingNumber
{
public:
    virtual ~IFastWindingNumber() = default;
    /// <summary>
    /// calculates winding numbers for a vector of points
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
    /// calculates winding numbers for each point in a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="minCoord">minimal coordinates of grid points</param>
    /// <param name="voxelSize">size of voxel</param>
    /// <param name="gridToMeshXf">transform from grid to mesh</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    virtual VoidOrErrStr calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb = {} ) = 0;

    /// <summary>
    /// calculates distances and winding numbers for each point in a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting signed distances, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="minCoord">minimal coordinates of grid points</param>
    /// <param name="voxelSize">size of voxel</param>
    /// <param name="gridToMeshXf">transform from grid to mesh</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    virtual VoidOrErrStr calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, float maxDistSq, float minDistSq, ProgressCallback cb ) = 0;

    /// <summary>
    /// returns amount of required memory for calcFromVector operation
    /// does not take into account size of output vector
    /// </summary>
    /// <param name="inputSize">size of input vector</param>
    virtual size_t fromVectorHeapBytes( size_t inputSize ) const = 0;

    /// <summary>
    /// returns amount of required memory for calcSelfIntersections operation
    /// does not take into account size of output FaceBitSet
    /// </summary>
    /// <param name="mesh">input mesh</param>
    virtual size_t selfIntersectionsHeapBytes( const Mesh& mesh ) const = 0;

    /// <summary>
    /// returns amount of required memory for calcFromGrid and calcFromGridWithDistances operation
    /// does not take into account size of output vector
    /// </summary>
    /// <param name="dims">dimensions of original grid</param>
    virtual size_t fromGridHeapBytes( const Vector3i& dims ) const = 0;
};

/// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
/// \ingroup AABBTreeGroup
class [[nodiscard]] FastWindingNumber : public IFastWindingNumber
{
public:
    /// constructs this from AABB tree of given mesh;
    /// this remains valid only if tree is valid
    [[nodiscard]] MRMESH_API FastWindingNumber( const Mesh & mesh );
    /// compute approximate winding number at \param q;
    /// \param beta determines the precision of the approximation: the more the better, recommended value 2 or more;
    /// if distance from q to the center of some triangle group is more than beta times the distance from the center to most distance triangle in the group then we use approximate formula
    /// \param skipFace this triangle (if it is close to \param q) will be skipped from summation
    [[nodiscard]] MRMESH_API float calc( const Vector3f & q, float beta, FaceId skipFace = {} ) const;
    /// <summary>
    /// calculates winding numbers for a vector of points
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="points">incoming points</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    /// <param name="skipFace">this triangle (if it is close to `q`) will be skipped from summation</param>
    MRMESH_API void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) override;
    /// <summary>
    /// calculates winding numbers for all centers of mesh's triangles. if winding number is less than 0 or greater then 1, that face is marked as self-intersected
    /// </summary>
    /// <param name="res">resulting bit set</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    MRMESH_API bool calcSelfIntersections( FaceBitSet& res, float beta, ProgressCallback cb ) override;
    /// <summary>
    /// calculates winding numbers for each point in a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting winding numbers, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="minCoord">minimal coordinates of grid points</param>
    /// <param name="voxelSize">size of voxel</param>
    /// <param name="gridToMeshXf">transform from grid to mesh</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    MRMESH_API VoidOrErrStr calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb ) override;

    /// calculates distances and winding numbers at \param p
    /// \param beta determines the precision of the approximation: the more the better, recommended value 2 or more;
    /// if distance from p to the center of some triangle group is more than beta times the distance from the center to most distance triangle in the group then we use approximate formula
    /// \param maxDistSq - maximum possible distance
    /// \param minDistSq - minimum possible distance
    MRMESH_API float calcWithDistances( const Vector3f& p, float beta, float maxDistSq, float minDistSq );
    /// <summary>
    /// calculates distances and winding numbers for each point in a three-dimensional grid
    /// </summary>
    /// <param name="res">resulting signed distances, will be resized automatically</param>
    /// <param name="dims">dimensions of the grid</param>
    /// <param name="minCoord">minimal coordinates of grid points</param>
    /// <param name="voxelSize">size of voxel</param>
    /// <param name="gridToMeshXf">transform from grid to mesh</param>
    /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
    MRMESH_API VoidOrErrStr calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, float maxDistSq, float minDistSq, ProgressCallback cb ) override;
    
    /// <summary>
    /// returns amount of required memory for calcFromVector operation
    /// does not take into account size of output vector
    /// </summary>
    /// <param name="inputSize">size of input vector</param>
    MRMESH_API virtual size_t fromVectorHeapBytes( size_t inputSize ) const override;

    /// <summary>
    /// returns amount of required memory for calcSelfIntersections operation
    /// does not take into account size of output FaceBitSet
    /// </summary>
    /// <param name="mesh">input mesh</param>
    MRMESH_API virtual size_t selfIntersectionsHeapBytes( const Mesh& mesh ) const override;

    /// <summary>
    /// returns amount of required memory for calcFromGrid and calcFromGridWithDistances operation
    /// does not take into account size of output vector
    /// </summary>
    /// <param name="dims">dimensions of original grid</param>
    MRMESH_API virtual size_t fromGridHeapBytes( const Vector3i& dims ) const override;

private:
    const Mesh & mesh_;
    const AABBTree & tree_;
    Dipoles dipoles_;
};

} // namespace MR
