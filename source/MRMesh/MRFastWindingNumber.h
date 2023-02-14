#pragma once

#include "MRAABBTree.h"
#include <array>

namespace MR
{

/// three vector3-coordinates describing a triangle geometry
using ThreePoints = std::array<Vector3f, 3>;

/// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
/// \ingroup AABBTreeGroup
class [[nodiscard]] FastWindingNumber
{
public:
    using NodeId = AABBTree::NodeId;
    /// constructs this from AABB tree of given mesh;
    /// this remains valid only if tree is valid
    [[nodiscard]] MRMESH_API FastWindingNumber( const Mesh & mesh );
    /// compute approximate winding number at \param q;
    /// \param beta determines the precision of the approximation: the more the better, recommended value 2 or more;
    /// if distance from q to the center of some triangle group is more than beta times the distance from the center to most distance triangle in the group then we use approximate formula
    [[nodiscard]] float calc( const Vector3f & q, float beta ) const;

private:
    struct Dipole
    {
        Vector3f areaPos;
        float area = 0;
        Vector3f dirArea;
        float rr = 0; // maximum squared distance from pos to any corner of the bounding box
        [[nodiscard]] Vector3f pos() const { return area > 0 ? areaPos / area : areaPos; }
        /// returns true if this dipole is good approximation for a point \param q
        [[nodiscard]] bool goodApprox( const Vector3f & q, float beta ) const { return ( q - pos() ).lengthSq() > sqr( beta ) * rr; }
        /// contribution of this dipole to the winding number at point \param q
        [[nodiscard]] float w( const Vector3f & q ) const;
        /// contribution of this dipole to the winding number at point \param q,
        /// considering that it corresponds to a single triangle with given vertex coordinates, which is subdivided automatically to reach desired beta-precision
        [[nodiscard]] float wSubdiv( const Vector3f & q, float beta, const ThreePoints & tri ) const;
    };
    static_assert( sizeof( Dipole ) == 8 * sizeof( float ) );
    using Dipoles = Vector<Dipole, NodeId>;
    const Mesh & mesh_;
    const AABBTree & tree_;
    Dipoles dipoles_;
};

} // namespace MR
