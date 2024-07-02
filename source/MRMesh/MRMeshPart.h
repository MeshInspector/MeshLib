#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)
/// \ingroup MeshAlgorithmGroup
struct MeshPart
{
    const Mesh & mesh;
    const FaceBitSet * region = nullptr; // nullptr here means whole mesh

    MeshPart( const Mesh & m, const FaceBitSet * bs = nullptr ) noexcept : mesh( m ), region( bs ) { }

    // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
    MeshPart(const MeshPart &other) noexcept = default;
    MeshPart &operator=(const MeshPart &other) noexcept
    {
        if (this != &other)
        {
            // In modern C++ the result doesn't need to be `std::launder`ed, right?
            this->~MeshPart();
            ::new((void *)this) MeshPart(other);
        }
        return *this;
    }
};

} // namespace MR
