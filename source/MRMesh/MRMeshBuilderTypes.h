#pragma once

#include "MRId.h"
#include <array>

namespace MR
{

namespace MeshBuilder
{

/// three vector3-coordinates describing a triangle
using ThreePoints = std::array<Vector3f, 3>;

/// mesh triangle represented by its three vertices and by its face ID
struct Triangle
{
    Triangle() noexcept = default;
    Triangle( VertId a, VertId b, VertId c, FaceId f ) : f(f) { v[0] = a; v[1] = b; v[2] = c; }
    VertId v[3];
    FaceId f;

    bool operator==( const Triangle& other )const
    {
        return f == other.f && v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
    }
};

} //namespace MeshBuilder

} //namespace MR
