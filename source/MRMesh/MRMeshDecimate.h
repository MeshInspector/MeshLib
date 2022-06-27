#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRConstants.h"
#include <climits>
#include <functional>

namespace MR
{
/**
 * \defgroup DecimateGroup Decimate overview
 * \brief This chapter represents documentation about mesh decimation
 */

/// Defines the order of edge collapses inside Decimate algorithm
enum DecimateStrategy
{
    MinimizeError,    // the next edge to collapse will be the one that introduced minimal error to the surface
    ShortestEdgeFirst // the next edge to collapse will be the shortest one
};

/**
 * \struct MR::DecimateSettings
 * \brief Parameters structure for MR::decimateMesh
 * \ingroup DecimateGroup
 * 
 * \sa \ref decimateMesh
 */
struct DecimateSettings
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
    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos = true;
    /// Limit on the number of deleted vertices
    int maxDeletedVertices = INT_MAX;
    /// Limit on the number of deleted faces
    int maxDeletedFaces = INT_MAX;
    /// Region on mesh to be decimated, it is updated during the operation
    FaceBitSet * region = nullptr;
    /// Whether to allow collapsing edges having at least one vertex on (region) boundary
    bool touchBdVertices = true;
    /**
     * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives the edge being collapsed: its destination vertex will disappear,
     * and its origin vertex will get new position (provided as the second argument) after collapse;
     * If the callback returns false, then the collapse is prohibited
     */
    std::function<bool( EdgeId edgeToCollapse, const Vector3f & newEdgeOrgPos)> preCollapse;
    /**
     * \brief The user can provide this optional callback for adjusting error introduced by this
     * edge collapse and the collapse position.
     * \details On input the callback gets the squared error and position computed by standard means,
     * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
     * This callback can be called from many threads in parallel and must be thread-safe.
     * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
     */
    std::function<void( UndirectedEdgeId ue, float & collapseErrorSq, Vector3f & collapsePos )> adjustCollapse;
    /**
     * \brief  If not null, then
     * on input: if the vector is not empty then it is takes for initialization instead of form computation for all vertices;
     * on output: quadratic form for each remaining vertex is returned there
     */
    Vector<QuadraticForm3f, VertId> * vertForms = nullptr;
    ///  whether to pack mesh at the end
    bool packMesh = false;
    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback = {};
};

/**
 * \struct MR::DecimateResult
 * \brief Results of MR::decimateMesh
 * \ingroup DecimateGroup
 * 
 * \sa \ref decimateMesh
 * \sa \ref decimateParallelMesh
 * \sa \ref resolveMeshDegenerations
 */
struct DecimateResult
{
    int vertsDeleted = 0; ///< Number deleted verts. Same as the number of performed collapses
    int facesDeleted = 0; ///< Number deleted faces
    /// for DecimateStrategy::MinimizeError:
    ///    estimated distance deviation of decimated mesh from the original mesh
    /// for DecimateStrategy::ShortestEdgeFirst:
    ///    the shortest remaining edge in the mesh
    float errorIntroduced = 0;
    /// whether the algorithm was cancelled by the callback
    bool cancelled = true;
};

/**
 * \brief Collapse edges in mesh region according to the settings
 * \ingroup DecimateGroup
 * \details Have version for parallel computing - \ref decimateParallelMesh
 *
 * \image html decimate/decimate_before.png "Before" width = 350cm
 * \image html decimate/decimate_after.png "After" width = 350cm
 * 
 * \sa \ref decimateParallelMesh
 * \sa \ref resolveMeshDegenerations
 */ 
MRMESH_API DecimateResult decimateMesh( Mesh & mesh, const DecimateSettings & settings = {} );

/**
 * \brief Computes quadratic form at given vertex of the initial surface before decimation
 * \ingroup DecimateGroup
 */
MRMESH_API QuadraticForm3f computeFormAtVertex( const MeshPart & mp, VertId v, float stabilizer );

/**
 * \brief Resolves degenerate triangles in given mesh
 * \details This function performs decimation, so it can affect topology
 * \ingroup DecimateGroup
 * \return true if the mesh has been changed
 * 
 * \sa \ref decimateMesh
 */
MRMESH_API bool resolveMeshDegenerations( Mesh& mesh, int maxIters = 1, float maxDeviation = 0 );

struct RemeshSettings
{
    // the algorithm will try to keep the length of all edges close to this value,
    // splitting twice longer edges, and eliminating twice shorter edges
    float targetEdgeLen = 0.001f;
    /// Improves local mesh triangulation by doing edge flips if it does change dihedral angle more than on this value
    float maxAngleChangeAfterFlip = 30 * PI_F / 180.0f;
    /// This option in subdivision works best for natural surfaces, where all triangles are close to equilateral and have similar area,
    /// and no sharp edges in between
    bool useCurvature = false;
    /// Region on mesh to be changed, it is updated during the operation
    FaceBitSet * region = nullptr;
    ///  whether to pack mesh at the end
    bool packMesh = false;
};
// Splits too long and eliminates too short edges from the mesh
MRMESH_API void remesh( Mesh& mesh, const RemeshSettings & settings );

} //namespace MR
