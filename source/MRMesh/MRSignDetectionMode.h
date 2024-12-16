#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// how to determine the sign of distances from a mesh
enum class SignDetectionMode
{
    /// unsigned distance, useful for bidirectional `Shell` offset
    Unsigned,

    /// sign detection from OpenVDB library, which is good and fast if input geometry is closed
    OpenVDB,

    /// the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
    ProjectionNormal,

    /// ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh;
    /// this mode is slow, and it does NOT have CUDA acceleration at this moment
    WindingRule,

    /// computes robust winding number generalization with support of holes and self-intersections in mesh,
    /// it is the slowest sign detection mode, but it CAN be accelerated with CUDA if this mode activated e.g. in OffsetParameters.fwn
    HoleWindingRule
};

/// returns string representation of enum values
[[nodiscard]] MRMESH_API const char * asString( SignDetectionMode m );

/// how to determine the sign of distances from a mesh, short version including auto-detection
enum class SignDetectionModeShort
{
    Auto,              ///< automatic selection of the fastest method among safe options for the current mesh
    HoleWindingNumber, ///< detects sign from the winding number generalization with support for holes and self-intersections in mesh
    ProjectionNormal   ///< detects sign from the pseudonormal in closest mesh point, which is fast but unsafe in the presence of holes and self-intersections in mesh
};

} //namespace MR
