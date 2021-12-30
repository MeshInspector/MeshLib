#pragma once
#include "MRColor.h"
#include "MRVector2.h"
#include <vector>

namespace MR
{

// struct to hold Image data
struct Image
{
    std::vector<Color> pixels;
    Vector2i resolution;
};

}