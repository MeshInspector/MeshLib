#include "MRFeatureObject.h"

#include "json/value.h"

namespace MR
{


AllVisualizeProperties FeatureObject::getAllVisualizeProperties() const
{
    AllVisualizeProperties ret = VisualObject::getAllVisualizeProperties();
    getAllVisualizePropertiesForEnum<FeatureVisualizePropertyType>( ret );
    return ret;
}

const ViewportMask& FeatureObject::getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const
{
    if ( auto value = type.tryGet<FeatureVisualizePropertyType>() )
    {
        switch ( *value )
        {
            case FeatureVisualizePropertyType::Subfeatures:
                return subfeatureVisibility_;
            case FeatureVisualizePropertyType::_count: break; // MSVC warns if this is missing, despite `[[maybe_unused]]` on the `_count`.
        }
        assert( false && "Invalid enum." );
        return visibilityMask_;
    }
    else
    {
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void FeatureObject::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    // append base type
    root["Type"].append( VisualObject::TypeName() );

    root["SubfeatureVisibility"] = subfeatureVisibility_.value();
}

void FeatureObject::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    if ( const auto& subfeatureVisibilityJson = root["SubfeatureVisibility"]; subfeatureVisibilityJson.isUInt() )
        subfeatureVisibility_ = ViewportMask( subfeatureVisibilityJson.asUInt() );
}

void FeatureObject::setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos )
{
    VisualObject::setAllVisualizeProperties_( properties, pos );
    setAllVisualizePropertiesForEnum<FeatureVisualizePropertyType>( properties, pos );
}

}
