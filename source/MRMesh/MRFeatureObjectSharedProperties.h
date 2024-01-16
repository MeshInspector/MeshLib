#pragma once
#include "MRMeshFwd.h"
#include <variant>

namespace MR
{

using FeatureObjectsSettersVariant = std::variant<float, Vector3f >;

// FeatureObjectSharedProperty struct is designed to represent a shared property of a feature object, enabling the use of generalized getter and setter methods for property manipulation.
// propertyName: A string representing the name of the property.
// getter : A std::function encapsulating a method with no parameters that returns a FeatureObjectsSettersVariant.This allows for a generic way to retrieve the value of the property.
// setter : A std::function encapsulating a method that takes a FeatureObjectsSettersVariant as a parameter and returns void.This function sets the value of the property.
// The templated constructor of this struct takes the property name, pointers to the getter and setter member functions, and a pointer to the object( obj ).
// The constructor initializes the propertyName and uses lambdas to adapt the member function pointers into std::function objects that conform to the expected 
// getter and setter signatures.The getter lambda invokes the getter method on the object, and the setter lambda ensures the correct variant type is passed before 
// invoking the setter method.

struct FeatureObjectSharedProperty {
    std::string propertyName;
    std::function<FeatureObjectsSettersVariant()> getter;
    std::function<void( FeatureObjectsSettersVariant )> setter;

    template <typename T, typename C>
    FeatureObjectSharedProperty( std::string name, T( C::* m_getter )( ) const, void ( C::* m_setter )( const T& ), C* obj )
        : propertyName( std::move( name ) ),
        getter( [obj, m_getter] () -> FeatureObjectsSettersVariant
    {
        return std::invoke( m_getter, obj );
    } ),
        setter( [obj, m_setter] ( FeatureObjectsSettersVariant v )
    {
        assert( std::holds_alternative<T>( v ) );
        if ( std::holds_alternative<T>( v ) )
        {
            std::invoke( m_setter, obj, std::get<T>( v ) );
        }
    } )
    {};
};

using FeatureObjectSharedProperties = std::vector<FeatureObjectSharedProperty>;


struct  FeatureObjectWithSharedProperties {
public:
    FeatureObjectWithSharedProperties( void ) noexcept = default;
    FeatureObjectWithSharedProperties( const FeatureObjectWithSharedProperties& ) noexcept = default;
    FeatureObjectWithSharedProperties( FeatureObjectWithSharedProperties&& ) noexcept = default;
    FeatureObjectWithSharedProperties& operator = ( FeatureObjectWithSharedProperties&& ) noexcept = default;
    virtual ~FeatureObjectWithSharedProperties() = default;

    virtual FeatureObjectSharedProperties getAllSharedProperties( void ) = 0;
};

}