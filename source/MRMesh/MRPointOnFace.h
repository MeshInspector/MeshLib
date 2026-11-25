#pragma once

#include "MRVector3.h"
#include "MRTriPoint.h"
#include "MRId.h"

#include <iosfwd>

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

/// a point on mesh given both by its position and by its barycentric coordinates
struct PointOnFaceBary
{
    /// the point on mesh (in MeshProjectionResult: the closest point, transformed by xf if it is given), and the triangle containing it
    PointOnFace proj;

    /// barycentric coordinates of proj relative to the vertices of proj.face in the order of MeshTopology::getTriVerts( proj.face )
    TriPointf tp;

    /// check for validity, otherwise the point was not found
    [[nodiscard]] bool valid() const { return proj.valid(); }
    [[nodiscard]] explicit operator bool() const { return proj.valid(); }

    /// returns the point as MeshTriPoint relative to topology.edgeWithLeft( proj.face ), or invalid MeshTriPoint if the point was not found
    [[nodiscard]] MRMESH_API MeshTriPoint mtp( const MeshTopology & topology ) const;
};

} //namespace MR
