#pragma once

#include "MRMeshDecimate.h"

namespace MR
{

/**
 * \struct MR::DecimateParallelSettings
 * \brief Parameters structure for MR::decimateParallelMesh
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
    ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
    float maxError = 0.001f;
    /// Edges longer than this value will not be collapsed (but they can appear after collapsing of shorter ones)
    float maxEdgeLen = 1;
    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio = 20;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 0.001f;
    /// Region on mesh to be decimated, it is updated during the operation
    FaceBitSet * region = nullptr;
    /// Subdivides mesh on given number of parts to process them in parallel
    int subdivideParts = 32;
    /**
     * \brief  The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives both vertices of the edge being collapsed: v1 will disappear,
     * and v0 will get new position (provided as the third argument) after collapse;
     * If the callback returns false, then the collapse is prohibited;
     * \note This callback will be called from parallel threads when they process subparts
     */
    std::function<bool( VertId v0, VertId v1, const Vector3f & newV0Pos )> preCollapse;
    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback = {};
};

/**
 * \brief Collapse edges in mesh region according to the settings
 * \ingroup DecimateGroup
 * \details Analog of decimateMesh for parallel computing. If accuracy is preferable to speed, use \ref decimateMesh.
 * 
 * \sa \ref decimateMesh
 */
MRMESH_API DecimateResult decimateParallelMesh( Mesh & mesh, const DecimateParallelSettings & settings = {} );

} //namespace MR
