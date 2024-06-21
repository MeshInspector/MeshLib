#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

enum MRSignDetectionMode
{
    MRSignDetectionModeUnsigned = 0,
    MRSignDetectionModeOpenVDB,
    MRSignDetectionModeProjectionNormal,
    MRSignDetectionModeWindingRule,
    MRSignDetectionModeHoleWindingRule
};

MR_EXTERN_C_END