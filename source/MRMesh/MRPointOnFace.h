#pragma once

#include "MRVector3.h"
#include "MRId.h"

namespace MR
{

// point located on some mesh face
struct PointOnFace
{
    FaceId face;
    Vector3f point;
};

} //namespace MR
