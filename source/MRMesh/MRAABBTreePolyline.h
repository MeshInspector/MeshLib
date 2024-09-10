#pragma once

#include "MRAABBTreeBase.h"
#include "MRVector.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

template<typename V>
struct PolylineTraits;

template<>
struct PolylineTraits<Vector2f>
{
    using Polyline = MR::Polyline2;
};

template<>
struct PolylineTraits<Vector3f>
{
    using Polyline = MR::Polyline3;
};

/// bounding volume hierarchy for line segments
template<typename V>
class AABBTreePolyline : public AABBTreeBase<LineTreeTraits<V>>
{
    using Base = AABBTreeBase<LineTreeTraits<V>>;

public:
    using typename Base::Traits;
    using typename Base::Node;
    using typename Base::NodeVec;

public:
    /// creates tree for given polyline
    MRMESH_API explicit AABBTreePolyline( const typename PolylineTraits<V>::Polyline & polyline );

    /// creates tree for selected edges on the mesh (only for 3d tree)
    MRMESH_API AABBTreePolyline( const Mesh& mesh, const UndirectedEdgeBitSet & edgeSet ) MR_REQUIRES_IF_SUPPORTED( V::elements == 3 );

    AABBTreePolyline() = default;
    AABBTreePolyline( AABBTreePolyline && ) noexcept = default;
    AABBTreePolyline & operator =( AABBTreePolyline && ) noexcept = default;

private:
    /// make copy constructor unavailable for the public to avoid unnecessary copies
    AABBTreePolyline( const AABBTreePolyline & ) = default;

    /// make assign operator unavailable for the public to avoid unnecessary copies
    AABBTreePolyline & operator =( const AABBTreePolyline & ) = default;

    friend class UniqueThreadSafeOwner<AABBTreePolyline>;

    using Base::nodes_;
};

/// \}

} // namespace MR
