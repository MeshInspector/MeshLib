#include "MREnums.h"
#include <cassert>

namespace MR
{

const char * asString( ColoringType ct )
{
    switch ( ct )
    {
    case ColoringType::SolidColor:
        return "SolidColor";
    case ColoringType::PrimitivesColorMap:
        return "PrimitivesColorMap";
    case ColoringType::VertsColorMap:
        return "VertsColorMap";
    default:
        assert( false );
        return "";
    }
}

} //namespace MR
