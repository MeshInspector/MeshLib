#pragma once

#include "MRAABBTree.h"
#include <array>
#include <string>

namespace MR
{
class IFastWindingNumber
{
protected:
    const Mesh& mesh_;

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

    using NodeId = AABBTree::NodeId;
    static_assert( sizeof( Dipole ) == 8 * sizeof( float ) );
    using Dipoles = Vector<Dipole, NodeId>;

public:
    IFastWindingNumber( const Mesh& mesh )
    : mesh_( mesh )
    {}
    virtual ~IFastWindingNumber() = default;

    virtual void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) = 0;
    virtual void  calcSelfIntersections( FaceBitSet& res, float beta ) = 0;
    virtual std::vector<std::string>  calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta ) = 0;

    static MRMESH_API void calcDipoles( Dipoles& dipoles, const AABBTree& tree, const Mesh& mesh );
};
/// three vector3-coordinates describing a triangle geometry
using ThreePoints = std::array<Vector3f, 3>;

/// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
/// \ingroup AABBTreeGroup
class [[nodiscard]] FastWindingNumber : public IFastWindingNumber
{
public:
    /// constructs this from AABB tree of given mesh;
    /// this remains valid only if tree is valid
    [[nodiscard]] MRMESH_API FastWindingNumber( const Mesh & mesh );
    /// compute approximate winding number at \param q;
    /// \param beta determines the precision of the approximation: the more the better, recommended value 2 or more;
    /// if distance from q to the center of some triangle group is more than beta times the distance from the center to most distance triangle in the group then we use approximate formula
    /// \param skipFace this triangle (if it is close to \param q) will be skipped from summation
    [[nodiscard]] MRMESH_API float calc( const Vector3f & q, float beta, FaceId skipFace = {} ) const;

    MRMESH_API void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) override;
    MRMESH_API void calcSelfIntersections( FaceBitSet& res, float beta ) override;
    MRMESH_API std::vector<std::string> calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta ) override;
    
private:
    const AABBTree & tree_;
    Dipoles dipoles_;
};

} // namespace MR
