#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)
/// \ingroup MeshAlgorithmGroup
template<typename RegionTag>
struct MeshRegion
{
    const Mesh& mesh;
    const TaggedBitSet<RegionTag>* region = nullptr; // nullptr here means whole mesh

    MeshRegion( const Mesh& m, const TaggedBitSet<RegionTag>* bs = nullptr ) noexcept : mesh( m ), region( bs )
    {}

    // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
    MeshRegion( const MeshRegion& other ) noexcept = default;
    MeshRegion& operator=( const MeshRegion& other ) noexcept
    {
        if ( this != &other )
        {
            // In modern C++ the result doesn't need to be `std::launder`ed, right?
            this->~MeshRegion();
            ::new( ( void* )this ) MeshRegion( other );
        }
        return *this;
    }
};

} // namespace MR
