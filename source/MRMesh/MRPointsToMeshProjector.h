#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include <float.h>
#include <string>

namespace MR
{
/// Abstract class, computes the closest point on mesh to each of given points. Pure virtual functions must be implemented
class IPointsToMeshProjector
{
protected:
public:
    virtual ~IPointsToMeshProjector() = default;    
    /// Updates all data related to the referencing mesh
    virtual void updateMeshData( const Mesh* mesh ) = 0;
    /// Computes the closest point on mesh to each of given points
    virtual void findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, 
                                  const AffineXf3f* worldXf = nullptr, const AffineXf3f* worldRefXf = nullptr,
                                  float upDistLimitSq = FLT_MAX, float loDistLimitSq = 0.0f ) = 0;

    /// Returns amount of memory needed to compute projections
    virtual size_t projectionsHeapBytes( size_t numProjections ) const = 0;
};
/// Computes the closest point on mesh to each of given points on CPU
class MRMESH_CLASS PointsToMeshProjector : public IPointsToMeshProjector
{
    const Mesh* mesh_{ nullptr };
public:
    /// update all data related to the referencing mesh
    MRMESH_API virtual void updateMeshData( const Mesh* mesh ) override;
    /// <summary>
    /// Computes the closest point on mesh to each of given points    
    /// </summary>
    /// <param name="result">vector pf projections</param>
    /// <param name="points">vector of points to project</param>
    /// <param name="objXf">transform applied to points</param>
    /// <param name="refObjXf">transform applied to referencing mesh</param>
    /// <param name="upDistLimitSq">maximal squared distance from point to mesh</param>
    /// <param name="loDistLimitSq">minimal squared distance from point to mesh</param>
    MRMESH_API virtual void findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, 
                                             const AffineXf3f* objXf, const AffineXf3f* refObjXf,
                                             float upDistLimitSq, float loDistLimitSq ) override;

    /// Returns amount of additional memory needed to compute projections
    MRMESH_API virtual size_t projectionsHeapBytes( size_t numProjections ) const override;
};

}