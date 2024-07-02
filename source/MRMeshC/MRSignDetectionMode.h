#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef enum MRSignDetectionMode
{
    MRSignDetectionModeUnsigned = 0,
    MRSignDetectionModeOpenVDB,
    MRSignDetectionModeProjectionNormal,
    MRSignDetectionModeWindingRule,
    MRSignDetectionModeHoleWindingRule
} MRSignDetectionMode;

MR_EXTERN_C_END