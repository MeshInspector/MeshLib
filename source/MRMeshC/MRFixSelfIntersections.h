#pragma once
#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// Fix method
typedef enum MRFixSelfIntersectionsMethod
{
    /// Relax mesh around self-intersections
    MRFixSelfIntersectionsMethodRelax,
    /// Cut and re-fill regions around self-intersections (may fall back to `Relax`)
    MRFixSelfIntersectionsMethodCutAndFill
} MRFixSelfIntersectionsMethod;

/// Setting set for mesh self-intersections fix
typedef struct MRFixSelfIntersectionsSettings
{
    /// If true then count touching faces as self-intersections
    bool touchIsIntersection;
    /// Fix method
    MRFixSelfIntersectionsMethod method;
    /// Maximum relax iterations
    int relaxIterations;
    /// Maximum expand count (edge steps from self-intersecting faces), should be > 0
    int maxExpand;
    /// Edge length for subdivision of holes covers (0.0f means auto)
    /// FLT_MAX to disable subdivision
    float subdivideEdgeLen;
    /// Callback function
    MRProgressCallback cb;
} MRFixSelfIntersectionsSettings;

/// creates a default instance
MRMESHC_API MRFixSelfIntersectionsSettings mrFixSelfIntersectionsSettingsNew(void);

/// Find all self-intersections faces component-wise
MRMESHC_API MRFaceBitSet* mrFixSelfIntersectionsGetFaces( const MRMesh* mesh, bool touchIsIntersection, MRProgressCallback cb, MRString** errorString );

/// Finds and fixes self-intersections per component:
MRMESHC_API void mrFixSelfIntersectionsFix( MRMesh* mesh, const MRFixSelfIntersectionsSettings* settings, MRString** errorString );

MR_EXTERN_C_END
