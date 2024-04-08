#include "MRObjectDimensionsEnum.h"

namespace MR
{

std::string_view toString( DimensionsVisualizePropertyType value )
{
    switch ( value )
    {
        case DimensionsVisualizePropertyType::diameter: return "Diameter";
        case DimensionsVisualizePropertyType::angle:    return "Angle";
        case DimensionsVisualizePropertyType::length:   return "Length";
        case DimensionsVisualizePropertyType::_count: break; // MSVC warns otherwise.
    }

    assert( false && "Invalid enum." );
    return {};
}

}
