#pragma once
#include "MRMeshFwd.h"
#include <variant>

namespace MR
{

using FeaturesPropertyTypesVariant = std::variant<float, Vector3f >;

// FeatureObjectSharedProperty struct is designed to represent a shared property of a feature object, enabling the use of generalized getter and setter methods for property manipulation.
// propertyName: A string representing the name of the property.
// getter : A std::function encapsulating a method with no parameters that returns a FeaturesPropertyTypesVariant.This allows for a generic way to retrieve the value of the property.
// setter : A std::function encapsulating a method that takes a FeaturesPropertyTypesVariant as a parameter and returns void.This function sets the value of the property.
// The templated constructor of this struct takes the property name, pointers to the getter and setter member functions, and a pointer to the object( obj ).
// The constructor initializes the propertyName and uses lambdas to adapt the member function pointers into std::function objects that conform to the expected 
// getter and setter signatures.The getter lambda invokes the getter method on the object, and the setter lambda ensures the correct variant type is passed before 
// invoking the setter method.

struct FeatureObjectSharedProperty {
    std::string propertyName;
    std::function<FeaturesPropertyTypesVariant()> getter;
    std::function<void( FeaturesPropertyTypesVariant )> setter;

    template <typename T, typename C, typename SetterFunc>
    FeatureObjectSharedProperty(
        std::string name,
        T( C::* m_getter )( ) const,
        SetterFunc m_setter,
        C* obj
    ) : propertyName( std::move( name ) ),
        getter( [obj, m_getter] () -> FeaturesPropertyTypesVariant
    {
        return std::invoke( m_getter, obj );
    } )
    {
        if constexpr ( ( std::is_same_v<SetterFunc, void ( C::* )( const T& )> )
            || ( std::is_same_v<SetterFunc, void ( C::* )( T )> ) )
        {
            setter = [obj, m_setter] ( FeaturesPropertyTypesVariant v )
            {
                assert( std::holds_alternative<T>( v ) );
                if ( std::holds_alternative<T>( v ) )
                {
                    std::invoke( m_setter, obj, std::get<T>( v ) );
                }
            };
        }
        else
        {
            static_assert( std::is_same_v<SetterFunc, T>, "Setter function signature unsupported" );
        }
    }
};

/// An interface class which allows feature objects to share setters and getters on their main properties, for convenient presentation in the UI
struct  FeatureObjectWithSharedProperties {
public:
    FeatureObjectWithSharedProperties( void ) noexcept = default;
    FeatureObjectWithSharedProperties( const FeatureObjectWithSharedProperties& ) noexcept = default;
    FeatureObjectWithSharedProperties( FeatureObjectWithSharedProperties&& ) noexcept = default;
    FeatureObjectWithSharedProperties& operator = ( FeatureObjectWithSharedProperties&& ) noexcept = default;
    virtual ~FeatureObjectWithSharedProperties() = default;

    /// Create and generate list of bounded getters and setters for the main properties of feature object, together with prop. name for display and edit into UI.
    virtual std::vector<FeatureObjectSharedProperty> getAllSharedProperties( void ) = 0;
};

}