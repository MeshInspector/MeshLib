#pragma once

#include "MRAABBTree.h"

namespace MR
{

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
    /// \param beta determines the precision of the approximation: the more the better
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
    };
    static_assert( sizeof( Dipole ) == 8 * sizeof( float ) );
    using Dipoles = Vector<Dipole, NodeId>;
    const AABBTree & tree_;
    Dipoles dipoles_;
};

} // namespace MR
