#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// Single oriented point or two oppositely charged points close together, representing a mesh part (one or more triangles)
/// https://www.dgp.toronto.edu/projects/fast-winding-numbers/fast-winding-numbers-for-soups-and-clouds-siggraph-2018-barill-et-al.pdf
struct Dipole
{
    Vector3f areaPos;
    float area = 0;
    Vector3f dirArea;
    float rr = 0; // maximum squared distance from pos to any corner of the bounding box
    [[nodiscard]] Vector3f pos() const
    {
        return area > 0 ? areaPos / area : areaPos;
    }
    /// returns true if this dipole is good approximation for a point \param q
    [[nodiscard]] bool goodApprox( const Vector3f& q, float beta ) const
    {
        return ( q - pos() ).lengthSq() > sqr( beta ) * rr;
    }
    /// contribution of this dipole to the winding number at point \param q
    [[nodiscard]] MRMESH_API float w( const Vector3f& q ) const;
};

static_assert( sizeof( Dipole ) == 8 * sizeof( float ) );

/// calculates dipoles for given mesh and AABB-tree
MRMESH_API void calcDipoles( Dipoles& dipoles, const AABBTree& tree, const Mesh& mesh );
[[nodiscard]] MRMESH_API Dipoles calcDipoles( const AABBTree& tree, const Mesh& mesh );

/// compute approximate winding number at \param q;
/// \param beta determines the precision of the approximation: the more the better, recommended value 2 or more;
/// if distance from q to the center of some triangle group is more than beta times the distance from the center to most distance triangle in the group then we use approximate formula
/// \param skipFace this triangle (if it is close to \param q) will be skipped from summation
[[nodiscard]] MRMESH_API float calcFastWindingNumber( const Dipoles& dipoles, const AABBTree& tree, const Mesh& mesh,
    const Vector3f & q, float beta, FaceId skipFace );

} //namespace MR
