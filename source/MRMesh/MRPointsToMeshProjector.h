#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include <string>

namespace MR
{
/// Abstract class, computes the closest point on mesh to each of given points. Pure virtual finctions must be implemented
class IPointsToMeshProjector
{
protected:
public:
    virtual ~IPointsToMeshProjector() = default;    
    /// update all data related to the referencing mesh
    virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) = 0;
    /// Computes the closest point on mesh to each of given points
    virtual void findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, 
                                  const AffineXf3f& worldXf, const AffineXf3f& worldRefXf, 
                                  float upDistLimitSq, float loDistLimitSq ) = 0;
};
/// Computes the closest point on mesh to each of given points on CPU
class MRMESH_CLASS PointsToMeshProjector : public IPointsToMeshProjector
{
    std::shared_ptr<const Mesh> mesh_;
    AffineXf3f xf_;
    const AffineXf3f* refXfPtr_ = nullptr;
public:
    /// update all data related to the referencing mesh
    MRMESH_API virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) override;
    /// <summary>
    /// Computes the closest point on mesh to each of given points    
    /// </summary>
    /// <param name="result">vector pf projections</param>
    /// <param name="points">vector of points to project</param>
    /// <param name="objXf">transform applied to points</param>
    /// <param name="refObjXf">transform applied to referencing mesh</param>
    /// <param name="upDistLimitSq">maximal squared distance from point to mesh</param>
    /// <param name="loDistLimitSq">minimal squared distance from point to mesh</param>
    /// <returns></returns>
    MRMESH_API virtual void findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, 
                                             const AffineXf3f& objXf, const AffineXf3f& refObjXf, 
                                             float upDistLimitSq, float loDistLimitSq ) override;
};

}