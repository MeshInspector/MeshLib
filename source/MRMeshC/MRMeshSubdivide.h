#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRSubdivideSettings
{
    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen;

    /// Maximum number of edge splits allowed
    int maxEdgeSplits;

    /// Improves local mesh triangulation by doing edge flips if it does not make too big surface deviation
    float maxDeviationAfterFlip;

    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
    float maxAngleChangeAfterFlip;

    /// If this value is less than FLT_MAX then edge flips will
    /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
    /// Unit: rad
    float criticalAspectRatioFlip;

    /// Region on mesh to be subdivided, it is updated during the operation
    const MRFaceBitSet * region;

    /// Edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
    MRUndirectedEdgeBitSet* notFlippable;

    /// New vertices appeared during subdivision will be added here
    MRVertBitSet * newVerts;

    /// If false do not touch border edges (cannot subdivide lone faces)\n
    /// use \ref MR::findRegionOuterFaces to find boundary faces
    bool subdivideBorder;

    /// The subdivision stops as soon as all triangles (in the region) have aspect ratio below or equal to this value
    float maxTriAspectRatio;

    /// An edge is subdivided only if both its left and right triangles have aspect ratio below or equal to this value.
    /// So this is a maximum aspect ratio of a triangle that can be split on two before Delone optimization.
    /// Please set it to a smaller value only if subdivideBorder==false, otherwise many narrow triangles can appear near border
    float maxSplittableTriAspectRatio;

    /// Puts new vertices so that they form a smooth surface together with existing vertices.
    /// This option works best for natural surfaces without sharp edges in between triangles
    bool smoothMode;

    /// In case of activated smoothMode, the smoothness is locally deactivated at the edges having
    /// dihedral angle at least this value
    float minSharpDihedralAngle;

    /// if true, then every new vertex will be projected on the original mesh (before smoothing)
    bool projectOnOriginalMesh;

    // TODO: beforeEdgeSplit;
    // TODO: onVertCreated;
    // TODO: onEdgeSplit;

    /// callback to report algorithm progress and cancel it by user request
    MRProgressCallback progressCallback;
} MRSubdivideSettings;

MRMESHC_API MRSubdivideSettings mrSubdivideSettingsNew( void );

/// splits edges in mesh region according to the settings;\n
/// \return The total number of edge splits performed
MRMESHC_API int mrSubdivideMesh( MRMesh* mesh, const MRSubdivideSettings* settings );

MR_EXTERN_C_END
