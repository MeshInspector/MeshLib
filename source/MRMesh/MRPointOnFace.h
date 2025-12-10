#pragma once

#include "MRVector3.h"
#include "MRId.h"

#include <iosfwd>
#if MR_PARSING_FOR_ANY_BINDINGS || MR_COMPILING_ANY_BINDINGS
#include <istream>
#include <ostream>
#endif

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

    MRMESH_API friend std::ostream& operator<<( std::ostream& s, const PointOnFace& pof );
    MRMESH_API friend std::istream& operator>>( std::istream& s, PointOnFace& pof );
};

} //namespace MR
