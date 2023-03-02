#pragma once

namespace MR
{

/// how to determine the sign of distances from a mesh
enum class SignDetectionMode
{
    Unsigned,         // unsigned distance, useful for bidirectional `Shell` offset
    ProjectionNormal, // the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
    WindingRule,      // ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh
    HoleWindingRule   // computes winding number generalization with support of holes in mesh, slower than WindingRule
};

} //namespace MR
