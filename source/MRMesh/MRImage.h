#pragma once

#include "MRColor.h"
#include "MRVector2.h"
#include "MRHeapBytes.h"
#include <vector>

namespace MR
{

/// struct to hold Image data
/// \ingroup BasicStructuresGroup
struct Image
{
    std::vector<Color> pixels;
    Vector2i resolution;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return MR::heapBytes( pixels ); }
};

}