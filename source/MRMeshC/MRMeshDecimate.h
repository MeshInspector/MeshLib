#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// Defines the order of edge collapses inside Decimate algorithm
typedef enum MRDecimateStrategy
{
    /// the next edge to collapse will be the one that introduced minimal error to the surface
    MRDecimateStrategyMinimizeError = 0,
    /// the next edge to collapse will be the shortest one
    MRDecimateStrategyShortestEdgeFirst
} MRDecimateStrategy;

/// parameters for \ref mrDecimateMesh
typedef struct MRDecimateSettings
{
    MRDecimateStrategy strategy;
    /// for DecimateStrategy::MinimizeError:
    ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
    /// for DecimateStrategy::ShortestEdgeFirst only:
    ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
    float maxError;
    /// Maximal possible edge length created during decimation
    float maxEdgeLen;
    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift;
    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio;
    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalTriAspectRatio;
    /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
    float tinyEdgeLength;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer;
    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos;
    /// Limit on the number of deleted vertices
    int maxDeletedVertices;
    /// Limit on the number of deleted faces
    int maxDeletedFaces;
    /// Region on mesh to be decimated, it is updated during the operation
    MRFaceBitSet* region;
    // TODO: notFlippable
    /// Whether to allow collapse of edges incident to notFlippable edges,
    /// which can move vertices of notFlippable edges unless they are fixed
    bool collapseNearNotFlippable;
    // TODO: edgesToCollapse
    // TODO: twinMap
    /// Whether to allow collapsing or flipping edges having at least one vertex on (region) boundary
    bool touchNearBdEdges;
    /// touchBdVerts=true: allow moving and eliminating boundary vertices during edge collapses;
    /// touchBdVerts=false: allow only collapsing an edge having only one boundary vertex in that vertex, so position and count of boundary vertices do not change;
    /// this setting is ignored if touchNearBdEdges=false
    bool touchBdVerts;
    // TODO: bdVerts
    /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
    /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
    float maxAngleChange;
    // TODO: preCollapse
    // TODO: adjustCollapse
    // TODO: onEdgeDel
    // TODO: vertForms
    /// whether to pack mesh at the end
    bool packMesh;
    /// callback to report algorithm progress and cancel it by user request
    MRProgressCallback progressCallback;
    /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
    /// unlike \ref mrDecimateParallelMesh it does not create copies of mesh regions, so may take less memory to operate;
    /// IMPORTANT: please call mrMeshPackOptimally before calling decimating with subdivideParts > 1, otherwise performance will be bad
    int subdivideParts;
    /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
    /// to eliminate small edges near the border of individual parts
    bool decimateBetweenParts;
    // TODO: partFaces
    /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
    int minFacesInPart;
} MRDecimateSettings;

/// initializes a default instance
MRMESHC_API MRDecimateSettings mrDecimateSettingsNew( void );

/// results of mrDecimateMesh
typedef struct MRDecimateResult
{
    /// Number deleted verts. Same as the number of performed collapses
    int vertsDeleted;
    /// Number deleted faces
    int facesDeleted;
    /// for DecimateStrategy::MinimizeError:
    ///    estimated distance deviation of decimated mesh from the original mesh
    /// for DecimateStrategy::ShortestEdgeFirst:
    ///    the shortest remaining edge in the mesh
    float errorIntroduced;
    /// whether the algorithm was cancelled by the callback
    bool cancelled;
} MRDecimateResult;

/// Collapse edges in mesh region according to the settings
MRMESHC_API MRDecimateResult mrDecimateMesh( MRMesh* mesh, const MRDecimateSettings* settings );

/// parameters for \ref mrResolveMeshDegenerations
typedef struct MRResolveMeshDegenSettings
{
    /// maximum permitted deviation from the original surface
    float maxDeviation;
    /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
    float tinyEdgeLength;
    /// Permit edge flips if it does not change dihedral angle more than on this value
    float maxAngleChange;
    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalAspectRatio;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer;
    /// degenerations will be fixed only in given region, which is updated during the processing
    MRFaceBitSet* region;
} MRResolveMeshDegenSettings;

/// initializes a default instance
MRMESHC_API MRResolveMeshDegenSettings mrResolveMeshDegenSettingsNew( void );

/// Resolves degenerate triangles in given mesh
/// This function performs decimation, so it can affect topology
/// \return true if the mesh has been changed
MRMESHC_API bool mrResolveMeshDegenerations( MRMesh* mesh, const MRResolveMeshDegenSettings* settings );

/// parameters for \ref mrRemesh
typedef struct MRRemeshSettings
{
    /// the algorithm will try to keep the length of all edges close to this value,
    /// splitting the edges longer than targetEdgeLen, and then eliminating the edges shorter than targetEdgeLen
    float targetEdgeLen;
    /// Maximum number of edge splits allowed during subdivision
    int maxEdgeSplits;
    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value
    float maxAngleChangeAfterFlip;
    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift;
    /// This option in subdivision works best for natural surfaces, where all triangles are close to equilateral and have similar area,
    /// and no sharp edges in between
    bool useCurvature;
    /// the number of iterations of final relaxation of mesh vertices;
    /// few iterations can give almost perfect uniformity of the vertices and edge lengths but deviate from the original surface
    int finalRelaxIters;
    /// if true prevents the surface from shrinkage after many iterations
    bool finalRelaxNoShrinkage;
    /// Region on mesh to be changed, it is updated during the operation
    MRFaceBitSet* region;
    // TODO: notFlippable
    /// whether to pack mesh at the end
    bool packMesh;
    /// if true, then every new vertex after subdivision will be projected on the original mesh (before smoothing);
    /// this does not affect the vertices moved on other stages of the processing
    bool projectOnOriginalMesh;
    // TODO: onEdgeSplit
    // TODO: onEdgeDel
    // TODO: preCollapse
    /// callback to report algorithm progress and cancel it by user request
    MRProgressCallback progressCallback;
} MRRemeshSettings;

/// initializes a default instance
MRMESHC_API MRRemeshSettings mrRemeshSettingsNew( void );

/// Splits too long and eliminates too short edges from the mesh
MRMESHC_API bool mrRemesh( MRMesh* mesh, const MRRemeshSettings* settings );

MR_EXTERN_C_END
