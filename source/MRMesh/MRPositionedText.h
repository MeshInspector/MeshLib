#pragma once
#include "MRPch/MRBindingMacros.h"
#include "MRVector3.h"
#include <string>

namespace MR
{

struct PositionedText
{
    PositionedText() = default;
    PositionedText( const std::string& text, const Vector3f& position ) : text{text}, position{position}{}

    std::string text;
    Vector3f position;

    bool operator==( const PositionedText& ) const = default;
};

using MeshLabel [[deprecated]] MR_BIND_IGNORE = PositionedText;

}
