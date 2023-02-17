#pragma once
#include "MRImage.h"
#include "MRVector4.h"
#include <vector>

namespace MR
{

/// \addtogroup BasicStructuresGroup
/// \{

struct MeshTexture : Image
{
    FilterType filter = FilterType::Discrete;
    WrapType wrap = WrapType::Clamp;
};

/// \}

}