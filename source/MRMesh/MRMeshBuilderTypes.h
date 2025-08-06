#pragma once

#include "MRId.h"
#include <array>

namespace MR
{

/// three vector3-coordinates describing a triangle geometry
using ThreePoints [[deprecated]] MR_BIND_IGNORE = std::array<Vector3f, 3>;

namespace MeshBuilder
{

/// mesh triangle represented by its three vertices and by its face ID
struct Triangle
{
    Triangle() noexcept = default;
    Triangle( VertId a, VertId b, VertId c, FaceId f ) : f(f) { v[0] = a; v[1] = b; v[2] = c; }
    ThreeVertIds v;
    FaceId f;

    bool operator==( const Triangle& other )const
    {
        return f == other.f && v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
    }
};

struct BuildSettings
{
    /// if region is given then on input it contains the faces to be added, and on output the faces failed to be added
    FaceBitSet * region = nullptr;

    /// this value to be added to every faceId before its inclusion in the topology
    int shiftFaceId = 0;

    /// whether to permit non-manifold edges in the resulting topology
    bool allowNonManifoldEdge = true;

    /// optional output: counter of skipped faces during mesh creation
    int* skippedFaceCount = nullptr;
};

// each face is surrounded by a closed contour of vertices [fistVertex, lastVertex)
struct VertSpan
{
    int firstVertex = 0;
    int lastVertex = 0;
};

} //namespace MeshBuilder

} //namespace MR
