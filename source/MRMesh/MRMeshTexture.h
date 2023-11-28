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
    FilterType filter = FilterType::Linear;
    WrapType wrap = WrapType::Clamp;
};

/// \}

}