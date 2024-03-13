#pragma once

#include "MRMesh/MRVisualObject.h"

namespace MR
{

// Inherits from a datamodel object, adding some visual property masks to it.
// `BaseObjectType` is the datamodel type to inherit from.
// `Properties...` is the list of properties to add. Each must be a value from a enum marked as `IsVisualizeMaskEnum`.
template <typename BaseObjectType, auto ...Properties>
requires ( IsVisualizeMaskEnum<decltype(Properties)>::value && ... )
class AddVisualProperties : public BaseObjectType
{
public:
    bool supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const override
    {
        return ( ( type.tryGet<decltype( Properties )>() == Properties ) || ... ) || BaseObjectType::supportsVisualizeProperty( type );
    }

    AllVisualizeProperties getAllVisualizeProperties() const override
    {
        AllVisualizeProperties ret = BaseObjectType::getAllVisualizeProperties();
        ret.reserve( ret.size() + sizeof...(Properties) );
        ( void( ret.push_back( this->getVisualizePropertyMask( Properties ) ) ), ... );
        return ret;
    }

    const ViewportMask& getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const override
    {
        const ViewportMask* ret = nullptr;
        std::size_t i = 0;
        (void)( ( type.tryGet<decltype( Properties )>() == Properties ? ( ret = &propertyMasks_[i], true ) : ( i++, false ) ) || ... );
        if ( ret )
            return *ret;
        else
            return BaseObjectType::getVisualizePropertyMask( type );
    }

protected:
    AddVisualProperties()
    {
        propertyMasks_.fill( ViewportMask::all() );
    }

    void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos ) override
    {
        BaseObjectType::setAllVisualizeProperties_( properties, pos );
        for ( std::size_t i = 0; i < sizeof...(Properties); i++ )
            propertyMasks_[i] = properties[pos++];
    }

    // Constructor sets this to all ones by default.
    std::array<ViewportMask, sizeof...(Properties)> propertyMasks_;
};

}
