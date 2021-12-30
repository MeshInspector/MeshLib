#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)
struct MeshPart
{
    const Mesh & mesh;
    const FaceBitSet * region = nullptr; // nullptr here means whole mesh

    MeshPart( const Mesh & m, const FaceBitSet * bs = nullptr ) noexcept : mesh( m ), region( bs ) { }
};

} //namespace MR
