#pragma once
#include "MRVector3.h"
#include <string>

namespace MR
{

struct MeshLabel
{
    MeshLabel() = default;
    MeshLabel( const std::string& text, const Vector3f& position ) : text{text}, position{position}{}

    std::string text;
    Vector3f position;
};

}
