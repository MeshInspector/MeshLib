#pragma once

#include "MRVector3.h"
#include "MRId.h"

namespace MR
{

/// a point located on some mesh's face
struct PointOnFace
{
    /// mesh's face containing the point
    FaceId face;

    /// a point of the mesh's face
    Vector3f point;

    /// check for validity, otherwise the point is not defined
    [[nodiscard]] bool valid() const { return face.valid(); }
    [[nodiscard]] explicit operator bool() const { return face.valid(); }
};

} //namespace MR
