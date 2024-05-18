#pragma once

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
    [[nodiscard]] float w( const Vector3f& q ) const;
};

static_assert( sizeof( Dipole ) == 8 * sizeof( float ) );

/// <summary>
/// calculates dipoles for given mesh and AABB-tree
/// </summary>
MRMESH_API void calcDipoles( Dipoles& dipoles, const AABBTree& tree, const Mesh& mesh );

} //namespace MR
