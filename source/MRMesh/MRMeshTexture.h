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
    enum class FilterType
    {
        Linear,
        Discrete
    } filter = FilterType::Discrete;

    enum class WrapType
    {
        Repeat,
        Mirror,
        Clamp
    } wrap = WrapType::Clamp;
};

/// Coordinates on texture 
/// \param u,v should be in range [0..1], otherwise result depends on wrap type of texture (no need to clamp it, it is done on GPU if wrap type is "Clamp" )
struct UVCoord
{
    float u{0.0f};
    float v{0.0f};
};

/// \}

}