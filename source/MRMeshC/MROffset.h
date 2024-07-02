#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRSignDetectionMode.h"

MR_EXTERN_C_BEGIN

typedef struct MROffsetParameters
{
    float voxelSize;
    MRProgressCallback callBack;
    MRSignDetectionMode signDetectionMode;
    // TODO: fwn
    bool memoryEfficient;
} MROffsetParameters;

MRMESHC_API MROffsetParameters mrOffsetParametersDefault( void );

MRMESHC_API MRMesh* mrOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrDoubleOffsetMesh( MRMeshPart mp, float offsetA, float offsetB, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrMcOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrMcShellMeshRegion( const MRMesh* mesh, const MRFaceBitSet* region, float offset, const MROffsetParameters* params, MRString** errorString );

typedef enum MRGeneralOffsetParametersMode
{
    MRGeneralOffsetParametersModeSmooth = 0,
    MRGeneralOffsetParametersModeStandard,
    MRGeneralOffsetParametersModeSharpening
} MRGeneralOffsetParametersMode;

typedef struct MRGeneralOffsetParameters
{
    // TODO: outSharpEdges
    float minNewVertDev;
    float maxNewRank2VertDev;
    float maxNewRank3VertDev;
    float maxOldVertPosCorrection;
    MRGeneralOffsetParametersMode mode;
} MRGeneralOffsetParameters;

MRMESHC_API MRGeneralOffsetParameters mrGeneralOffsetParametersDefault( void );

MRMESHC_API MRMesh* mrSharpOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MRMESHC_API MRMesh* mrGeneralOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MRMESHC_API MRMesh* mrThickenMesh( const MRMesh* mesh, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MR_EXTERN_C_END
