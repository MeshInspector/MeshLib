#pragma once

#include "MRPointOnFace.h"

namespace MR
{

// point located on either
// 1. face of ObjectMesh
// 2. line of ObjectLines
// 3. point of ObjectPoints
struct PointOnObject
{
    PointOnObject() {} //default ctor is required by Clang
    /// 3D location on the object in local coordinates
    Vector3f point;
    /// z buffer value
    float zBuffer{ 1.0f };
    /// to which primitive that point pertains
    union
    {
        int primId = -1;
        FaceId face;             //for ObjectMesh
        UndirectedEdgeId uedge;  //for ObjectLines
        VertId vert;             //for ObjectPoints
    };
    [[nodiscard]] operator PointOnFace() const { return { .face = face, .point = point }; }
};

} //namespace MR
