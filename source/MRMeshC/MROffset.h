#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRSignDetectionMode.h"

MR_EXTERN_C_BEGIN

typedef struct MRMESHC_CLASS MROffsetParameters
{
    float voxelSize;
    MRProgressCallback callBack;
    MRSignDetectionMode signDetectionMode;
    // TODO: fwn
    bool memoryEfficient;
} MROffsetParameters;

MRMESHC_API MROffsetParameters mrOffsetParametersDefault();

MRMESHC_API MRMesh* mrOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrDoubleOffsetMesh( MRMeshPart mp, float offsetA, float offsetB, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrMcOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrMcShellMeshRegion( const MRMesh* mesh, const MRFaceBitSet* region, float offset, const MROffsetParameters* params, MRString** errorString );

enum MRGeneralOffsetParametersMode
{
    MRGeneralOffsetParametersModeSmooth = 0,
    MRGeneralOffsetParametersModeStandard,
    MRGeneralOffsetParametersModeSharpening
};

typedef struct MRMESHC_CLASS MRGeneralOffsetParameters
{
    // TODO: outSharpEdges
    float minNewVertDev;
    float maxNewRank2VertDev;
    float maxNewRank3VertDev;
    float maxOldVertPosCorrection;
    MRGeneralOffsetParametersMode mode;
} MRGeneralOffsetParameters;

MRMESHC_API MRGeneralOffsetParameters mrGeneralOffsetParametersDefault();

MRMESHC_API MRMesh* mrSharpOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MRMESHC_API MRMesh* mrGeneralOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MRMESHC_API MRMesh* mrThickenMesh( const MRMesh* mesh, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MR_EXTERN_C_END
