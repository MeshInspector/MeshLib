#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

enum MRDecimateStrategy
{
    MRDecimateStrategyMinimizeError = 0,
    MRDecimateStrategyShortestEdgeFirst
};

typedef struct MRDecimateSettings
{
    MRDecimateStrategy strategy;
    float maxError;
    float maxEdgeLen;
    float maxBdShift;
    float maxTriangleAspectRatio;
    float criticalTriAspectRatio;
    float tinyEdgeLength;
    float stabilizer;
    bool optimizeVertexPos;
    int maxDeletedVertices;
    int maxDeletedFaces;
    MRFaceBitSet* region;
    // TODO: notFlippable
    // TODO: edgesToCollapse
    // TODO: touchBdVertices
    // TODO: bdVerts
    float maxAngleChange;
    // TODO: preCollapse
    // TODO: adjustCollapse
    // TODO: onEdgeDel
    // TODO: vertForms
    bool packMesh;
    MRProgressCallback progressCallback;
    int subdivideParts;
    bool decimateBetweenParts;
    // TODO: partFaces
    int minFacesInPart;
} MRDecimateSettings;

MRDecimateSettings mrDecimateSettingsDefault();

typedef struct MRDecimateResult
{
    int vertsDeleted;
    int facesDeleted;
    float errorIntroduced;
    bool cancelled;
} MRDecimateResult;

MRDecimateResult mrDecimateMesh( MRMesh* mesh, const MRDecimateSettings* settings );

typedef struct MRResolveMeshDegenSettings
{
    float maxDeviation;
    float tinyEdgeLength;
    float maxAngleChange;
    float criticalAspectRatio;
    float stabilizer;
    MRFaceBitSet* region;
} MRResolveMeshDegenSettings;

MRResolveMeshDegenSettings mrResolveMeshDegenSettingsDefault();

bool mrResolveMeshDegenerations( MRMesh* mesh, const MRResolveMeshDegenSettings* settings );

typedef struct MRRemeshSettings
{
    float targetEdgeLen;
    int maxEdgeSplits;
    float maxAngleChangeAfterFlip;
    float maxBdShift;
    bool useCurvature;
    int finalRelaxIters;
    bool finalRelaxNoShrinkage;
    MRFaceBitSet* region;
    // TODO: notFlippable
    bool packMesh;
    bool projectOnOriginalMesh;
    // TODO: onEdgeSplit
    // TODO: onEdgeDel
    // TODO: preCollapse
    MRProgressCallback progressCallback;
} MRRemeshSettings;

MRRemeshSettings mrRemeshSettingsDefault();

bool mrRemesh( MRMesh* mesh, const MRRemeshSettings* settings );

#ifdef __cplusplus
}
#endif
