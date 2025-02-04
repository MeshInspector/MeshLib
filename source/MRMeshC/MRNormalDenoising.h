#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef struct MRDenoiseViaNormalsSettings
{
    /// use approximated computation, which is much faster than precise solution
    bool fastIndicatorComputation;

    /// 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
    float beta;

    /// the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
    float gamma;

    /// the number of iterations to smooth normals and find creases; the more the better quality, but longer computation
    int normalIters;

    /// the number of iterations to update vertex coordinates from found normals; the more the better quality, but longer computation
    int pointIters;

    /// how much resulting points must be attracted to initial points (e.g. to avoid general shrinkage), must be > 0
    float guideWeight;

    /// if true then maximal displacement of each point during denoising will be limited
    bool limitNearInitial;

    /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
    float maxInitialDist;

    /// optionally returns creases found during smoothing
    MRUndirectedEdgeBitSet * outCreases;

    /// to get the progress and optionally cancel
    MRProgressCallback cb;
} MRDenoiseViaNormalsSettings;

MRMESHC_API MRDenoiseViaNormalsSettings mrDenoiseViaNormalsSettingsNew( void );

/// Reduces noise in given mesh,
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
MRMESHC_API void mrMeshDenoiseViaNormals( MRMesh* mesh, const MRDenoiseViaNormalsSettings* settings, MRString** errorString );

MR_EXTERN_C_END
