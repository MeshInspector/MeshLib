#pragma once

#include "MRMeshDecimate.h"

namespace MR
{

/**
 * \defgroup DecimateGroup Decimate overview
 * \brief This chapter represents documentation about mesh decimation
 */

/**
 * \struct MR::DecimateSettings
 * \brief Parameters structure for MR::decimateMesh
 * \ingroup DecimateGroup
 *
 * \sa \ref decimateMesh
 */
struct DecimateSettingsDouble
{
    DecimateStrategy strategy = DecimateStrategy::MinimizeError;

    /// for DecimateStrategy::MinimizeError:
    ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
    /// for DecimateStrategy::ShortestEdgeFirst only:
    ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
    double maxError = DBL_MAX;

    /// Maximal possible edge length created during decimation
    double maxEdgeLen = DBL_MAX;

    /// Maximal shift of a boundary during one edge collapse
    double maxBdShift = DBL_MAX;

    /// Maximal possible aspect ratio of a triangle introduced during decimation
    double maxTriangleAspectRatio = 20;

    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    double criticalTriAspectRatio = DBL_MAX;

    /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
    double tinyEdgeLength = -1;

    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    double stabilizer = 0.001f;

    /// if false, then quadratic error metric is equal to the sum of distances to the planes of original mesh triangles;
    /// if true, then the sum is weighted, and the weight is equal to the angle of adjacent triangle at the vertex divided on PI (to get one after summing all 3 vertices of the triangle)
    bool angleWeightedDistToPlane = false;

    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos = true;

    /// Limit on the number of deleted vertices
    int maxDeletedVertices = INT_MAX;

    /// Limit on the number of deleted faces
    int maxDeletedFaces = INT_MAX;

    /// Region on mesh to be decimated, it is updated during the operation
    FaceBitSet * region = nullptr;

    /// Edges specified by this bit-set will never be flipped, but they can be collapsed or replaced during collapse of nearby edges so it is updated during the operation
    UndirectedEdgeBitSet* notFlippable = nullptr;

    /// Whether to allow collapse of edges incident to notFlippable edges,
    /// which can move vertices of notFlippable edges unless they are fixed
    bool collapseNearNotFlippable = false;

    /// If pointer is not null, then only edges from here can be collapsed (and some nearby edges can disappear);
    /// the algorithm updates this map during collapses, removing or replacing elements
    UndirectedEdgeBitSet * edgesToCollapse = nullptr;

    /// if an edge present as a key in this map is flipped or collapsed, then same happens to the value-edge (with same collapse position);
    /// the algorithm updates this map during collapses, removing or replacing elements
    UndirectedEdgeHashMap * twinMap = nullptr;

    /// Whether to allow collapsing or flipping edges having at least one vertex on (region) boundary
    bool touchNearBdEdges = true;

    /// touchBdVerts=true: allow moving and eliminating boundary vertices during edge collapses;
    /// touchBdVerts=false: allow only collapsing an edge having only one boundary vertex in that vertex, so position and count of boundary vertices do not change;
    /// this setting is ignored if touchNearBdEdges=false
    bool touchBdVerts = true;

    /// if touchNearBdEdges=false or touchBdVerts=false then the algorithm needs to know about all boundary vertices;
    /// if the pointer is not null then boundary vertices detection is replaced with testing values in this bit-set
    const VertBitSet * bdVerts = nullptr;

    /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
    /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
    double maxAngleChange = -1;

    /**
     * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives the edge being collapsed: its destination vertex will disappear,
     * and its origin vertex will get new position (provided as the second argument) after collapse;
     * If the callback returns false, then the collapse is prohibited
     */
    PreCollapseCallback preCollapse;

    /**
     * \brief The user can provide this optional callback for adjusting error introduced by this
     * edge collapse and the collapse position.
     * \details On input the callback gets the squared error and position computed by standard means,
     * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
     * This callback can be called from many threads in parallel and must be thread-safe.
     * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
     */
    std::function<void( UndirectedEdgeId ue, double & collapseErrorSq, Vector3d & collapsePos )> adjustCollapse;

    /// this function is called each time edge (del) is deleted;
    /// if valid (rem) is given then dest(del) = dest(rem) and their origins are in different ends of collapsing edge, (rem) shall take the place of (del)
    std::function<void( EdgeId del, EdgeId rem )> onEdgeDel;

    /**
     * \brief  If not null, then vertex quadratic forms are stored there;
     * if on input the vector is not empty then initialization is skipped in favor of values from there;
     * on output: quadratic form for each remaining vertex is returned there
     */
    Vector<QuadraticForm3d, VertId> * vertForms = nullptr;

    ///  whether to pack mesh at the end
    bool packMesh = false;

    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback;

    /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
    /// IMPORTANT: please call mesh.packOptimally() before calling decimating with subdivideParts > 1, otherwise performance will be bad
    int subdivideParts = 1;

    /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
    /// to eliminate small edges near the border of individual parts
    bool decimateBetweenParts = true;

    /// if not null, then it contains the faces of each subdivision part on input, which must not overlap,
    /// and after decimation of all parts, the region inside each part is put here;
    /// decimateBetweenParts=true or packMesh=true are not compatible with this option
    std::vector<FaceBitSet> * partFaces = nullptr;

    /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
    int minFacesInPart = 0;
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
MRMESH_API DecimateResult decimateMeshDouble( Mesh & mesh, const DecimateSettingsDouble & settings = {} );

/**
 * \brief Computes quadratic form at given vertex of the initial surface before decimation
 * \ingroup DecimateGroup
 */
//[[nodiscard]] MRMESH_API QuadraticForm3d computeDoubleFormAtVertex( const MeshPart & mp, VertId v, double stabilizer, bool angleWeigted, const UndirectedEdgeBitSet * creases = nullptr );

/**
 * \brief Computes quadratic forms at every vertex of mesh part before decimation
 * \ingroup DecimateGroup
 */
//[[nodiscard]] MRMESH_API Vector<QuadraticForm3d, VertId> computeDoubleFormsAtVertices( const MeshPart & mp, double stabilizer, bool angleWeigted, const UndirectedEdgeBitSet * creases = nullptr );

} //namespace MR
