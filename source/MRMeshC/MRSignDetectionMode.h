#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// how to determine the sign of distances from a mesh
typedef enum MRSignDetectionMode
{
    /// unsigned distance, useful for bidirectional `Shell` offset
    MRSignDetectionModeUnsigned = 0,
    /// sign detection from OpenVDB library, which is good and fast if input geometry is closed
    MRSignDetectionModeOpenVDB,
    /// the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
    MRSignDetectionModeProjectionNormal,
    /// ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh
    MRSignDetectionModeWindingRule,
    /// computes winding number generalization with support of holes in mesh, slower than WindingRule
    MRSignDetectionModeHoleWindingRule
} MRSignDetectionMode;

MR_EXTERN_C_END
