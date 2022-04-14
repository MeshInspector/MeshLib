#pragma once

#include "MREMeshDecimate.h"

namespace MRE
{

/**
 * \struct MRE::DecimateParallelSettings
 * \brief Parameters structure for MRE::decimateParallelMesh
 * \ingroup DecimateGroup
 *
 * \sa \ref decimateParallelMesh
 */
struct DecimateParallelSettings
{  
    DecimateStrategy strategy = DecimateStrategy::MinimizeError;
    /// for DecimateStrategy::MinimizeError: 
    ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
    /// for DecimateStrategy::ShortestEdgeFirst only:
    ///   stop the decimation as soon as the longest edge in the mesh is greater than this value
    float maxError = 0.001f;
    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio = 20;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 0.001f;
    /// Subdivides mesh on given number of parts to process them in parallel
    int subdivideParts = 32;
    /**
     * \brief  The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives both vertices of the edge being collapsed: v1 will disappear,
     * and v0 will get new position (provided as the third argument) after collapse;
     * If the callback returns false, then the collapse is prohibited;
     * \note This callback will be called from parallel threads when they process subparts
     */
    std::function<bool( MR::VertId v0, MR::VertId v1, const MR::Vector3f & newV0Pos )> preCollapse;
};

/**
 * \brief Collapse edges in mesh region according to the settings
 * \ingroup DecimateGroup
 * \details Analog of decimateMesh for parallel computing. If accuracy is preferable to speed, use \ref decimateMesh.
 * 
 * \sa \ref decimateMesh
 */
MREALGORITHMS_API DecimateResult decimateParallelMesh( MR::Mesh & mesh, const DecimateParallelSettings & settings = {} );

} //namespace MRE
