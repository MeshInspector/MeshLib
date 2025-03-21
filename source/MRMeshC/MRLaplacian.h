#pragma once

#include "MRFillHoleNicely.h"
#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef enum MRLaplacianRememberShape
{
    MRLaplacianRememberShapeYes,  // true Laplacian mode when initial mesh shape is remembered and copied in apply
    MRLaplacianRememberShapeNo    // ignore initial mesh shape in the region and just position vertices smoothly in the region
} MRLaplacianRememberShape;

// Laplacian to smoothly deform a region preserving mesh fine details.
// How to use:
// 1. Initialize Laplacian for the region being deformed, here region properties are remembered.
// 2. Change positions of some vertices within the region and call fixVertex for them.
// 3. Optionally call updateSolver()
// 4. Call apply() to change the remaining vertices within the region
// Then steps 1-4 or 2-4 can be repeated.
typedef struct MRLaplacian MRLaplacian;

MRMESHC_API MRLaplacian* mrLaplacianNew( MRMesh* mesh );

MRMESHC_API void mrLaplacianFree( MRLaplacian* laplacian );

// initialize Laplacian for the region being deformed, here region properties are remembered and precomputed;
// \param freeVerts must not include all vertices of a mesh connected component
MRMESHC_API void mrLaplacianInit( MRLaplacian* laplacian, const MRVertBitSet* freeVerts, MREdgeWeights weights, MRLaplacianRememberShape rem );

// sets position of given vertex after init and it must be fixed during apply (THIS METHOD CHANGES THE MESH);
// \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
MRMESHC_API void mrLaplacianFixVertex( MRLaplacian* laplacian, MRVertId v, const MRVector3f* fixedPos, bool smooth );

// given fixed vertices, computes positions of remaining region vertices
MRMESHC_API void mrLaplacianApply( MRLaplacian* laplacian );

MR_EXTERN_C_END
