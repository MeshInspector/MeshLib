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

// Not 100% sure what's going on here, and whether the condition is 100% correct. `extern template` behind an `#if` is the only option that has worked for me.
// If you make this unconditional, you get undefined references in the python bindings, complaining about the functions defined inline in `MRAABBTreeBase.h` (sic!).
// In theory you just need to instantiate them in the .cpp file, but I couldn't figure out how to do that while preserving their dllexport-ness.
#if defined(_MSC_VER) && !defined(__clang__)
extern template class MRMESH_CLASS AABBTreeBase<LineTreeTraits<Vector2f>>;
extern template class MRMESH_CLASS AABBTreeBase<LineTreeTraits<Vector3f>>;
#endif

/// \}

} // namespace MR
