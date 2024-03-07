#pragma once

#include "MRMesh/MRVisualObject.h"

namespace MR
{

// For each of `{Radius,Angle,Length}VisualizePropertyType`, this holds the associated rendering properties.
template <typename T>
struct MeasurementPropertyParameters;

// Whether `T` is one of: `{Radius,Angle,Length}VisualizePropertyType`.
// Note: Clang-tidy 17 incorrect warns about this sizeof, even though this is a valid way of checking a type for completeness.
template <typename T>
concept MeasurementPropertyEnum = sizeof( MeasurementPropertyParameters<T> ) > 0; // NOLINT

template <MeasurementPropertyEnum T>
struct IsVisualizeMaskEnum<T> : std::true_type {};

// Those `{Radius,Angle,Length}VisualizePropertyType` enums have no constants, to support an arbitrary number of measurements.
// Cast the measurement index to the enum.

enum class MRMESH_CLASS RadiusVisualizePropertyType {};
template <> struct MeasurementPropertyParameters<RadiusVisualizePropertyType>
{
    // All of those are in local coordinates:

    // The center point.
    Vector3f center;

    // The length of this is the radius. This is also the preferred drawing direction relative to `center`.
    Vector3f radiusAsVector = Vector3f( 1, 0, 0 );

    // The preferred normal for non-spherical radiuses. The length is ignored, and this is automatically adjusted to be perpendicular to `radiusAsVector`.
    Vector3f normal = Vector3f( 0, 0, 1 );

    // Those can be serialized.
    struct VisualizationParams
    {
        // Whether we should draw this as a diameter instead of a radius.
        bool drawAsDiameter = false;

        // Whether this is a sphere radius, as opposed to circle/cylinder radius.
        bool isSpherical = false;

        // The visual leader line length multiplier, relative to the radius.
        // You're recommended to set a min absolute value for the resulting length when rendering.
        float visualLengthMultiplier = 2 / 3.f;
    };
    VisualizationParams vis;
};

enum class MRMESH_CLASS AngleVisualizePropertyType {};
template <> struct MeasurementPropertyParameters<AngleVisualizePropertyType>
{
    // All of those are in local coordinates:

    // The center point.
    Vector3f center;

    // The two rays.
    // Use the length of the shorter ray as the arc radius.
    Vector3f rays[2];

    // Those can be serialized.
    struct VisualizationParams
    {
        // Whether this is a conical angle. The middle line between the rays is preserved, but the rays themselves can be rotated.
        bool isConical = false;

        // Whether we should draw a ray from the center point to better visualize the angle. Enable this if there isn't already a line object there.
        bool shouldVisualizeRay[2]{};
    };
    VisualizationParams vis;
};

enum class MRMESH_CLASS LengthVisualizePropertyType {};
template <> struct MeasurementPropertyParameters<LengthVisualizePropertyType>
{
    // All of those are in local coordinates:

    // The points between which we're measuring.
    Vector3f points[2];

    // Those can be serialized.
    struct VisualizationParams
    {
        // Whether the distance should be displayed as a negative one.
        bool drawAsNegative = false;
    };
    VisualizationParams vis;
};

// A common base class for all classes using `ObjectWithMeasurements`.
struct IObjectWithMeasurements
{
    virtual ~IObjectWithMeasurements() = default;

    // Writes the parameters to `params`.
    // Validate the index first using `getVisualizePropertyMaskOpt( Kind( i ) )`.
    template <MeasurementPropertyEnum Kind>
    void getMeasurementParameters( std::size_t index, MeasurementPropertyParameters<Kind>& params ) const
    {
        getMeasurementParametersHelper_( index, typeid( Kind ), &params );
    }

protected:
    virtual void getMeasurementParametersHelper_( std::size_t index, std::type_index type, void *target ) const = 0;
};

// A mixin for data model objects, adding measurements.
// The type of `Kind` is one of: `{Radius,Angle,Length}VisualizePropertyType`.
// The value of `Kind` is now many measurements of this kind to add.
// `NextKinds...` are the same thing.
// All types in `Kind, NextKinds...` must be unique.
template <typename Base, auto Kind, auto ...NextKinds>
class ObjectWithMeasurements : public Base, public virtual IObjectWithMeasurements
{
    // Here `NextKinds...` is always empty. We have a separate specialization that handles the recursion.

    using KindType = decltype( Kind );

public:
    AllVisualizeProperties getAllVisualizeProperties() const override
    {
        AllVisualizeProperties ret = Base::getAllVisualizeProperties();
        ret.reserve( ret.size() + measurementVisibility_.size() );
        for ( std::size_t i = 0; i < measurementVisibility_.size(); i++ )
            ret.push_back( this->getVisualizePropertyMask( KindType( i ) ) );
        return ret;
    }

    const ViewportMask* getVisualizePropertyMaskOpt( AnyVisualizeMaskEnum type ) const override
    {
        if ( auto value = type.tryGet<KindType>() )
        {
            std::size_t index = std::size_t( *value );
            if ( index >= measurementVisibility_.size() )
                return nullptr;

            return &measurementVisibility_[index];
        }
        else
        {
            return Base::getVisualizePropertyMaskOpt( type );
        }
    }

protected:
    ObjectWithMeasurements()
    {
        measurementVisibility_.fill( ViewportMask::all() );
    }

    void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos ) override
    {
        Base::setAllVisualizeProperties_( properties, pos );
        for ( ViewportMask& mask : measurementVisibility_ )
            mask = properties[pos++];
    }

    void getMeasurementParametersHelper_( std::size_t index, std::type_index type, void *target ) const override
    {
        if ( type == typeid( KindType ) )
            *reinterpret_cast<MeasurementPropertyParameters<KindType> *>( target ) = getMeasurementParametersFor_( KindType( index ) );
        else if constexpr ( std::derived_from<Base, IObjectWithMeasurements> )
            Base::getMeasurementParametersHelper_( index, type, target );
        else
            assert( false && "This object doesn't hold this measurement type." );
    }

    // Must override this in the derived class!
    #ifdef __clang__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Woverloaded-virtual"
    #endif
    virtual MeasurementPropertyParameters<KindType> getMeasurementParametersFor_( KindType index ) const = 0;
    #ifdef __clang__
    #pragma clang diagnostic pop
    #endif

    // Constructor sets this to all ones by default.
    std::array<ViewportMask, std::size_t( Kind )> measurementVisibility_;
};

// This specialization folds for every `NextKinds...`.
template <typename Base, auto Kind, auto ...NextKinds>
requires ( sizeof...(NextKinds) > 0 )
class ObjectWithMeasurements<Base, Kind, NextKinds...>
    : public ObjectWithMeasurements<ObjectWithMeasurements<Base, NextKinds...>, Kind>
{
    static_assert( ( ( !std::is_same_v<decltype(Kind), decltype(NextKinds)> ) && ... ), "All measurement kinds must be unique." );
};

}
