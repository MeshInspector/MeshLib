#pragma once
#include "MRMeshFwd.h"
#include "MRVector2.h"
#include <array>

namespace MR
{
    MRMESH_API Mesh makePrism( const std::array<MR::Vector2f, 3>& points, float height = 1.0f );
}