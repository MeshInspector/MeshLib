#pragma once
#include "MRMesh/MRVisualObject.h"
#include "MRMeshFwd.h"

#include "MRVector3.h"

#include <cassert>
#include <variant>

namespace MR
{

using FeaturesPropertyTypesVariant = std::variant<float, Vector3f>;

class FeatureObject;

// Classifies `FeatureObjectSharedProperty`, mostly for informational purposes.
enum class FeaturePropertyKind
{
    position, // Position, normally Vector3f.
    linearDimension, // Length or size.
    direction, // Direction, normally Vector3f.
    angle, // Angle, normally float. Measure in radians.
    other,
};

// FeatureObjectSharedProperty struct is designed to represent a shared property of a feature object, enabling the use of generalized getter and setter methods for property manipulation.
// propertyName: A string representing the name of the property.
// getter : A std::function encapsulating a method with no parameters that returns a FeaturesPropertyTypesVariant.This allows for a generic way to retrieve the value of the property.
// setter : A std::function encapsulating a method that takes a FeaturesPropertyTypesVariant as a parameter and returns void.This function sets the value of the property.
// The templated constructor of this struct takes the property name, pointers to the getter and setter member functions, and a pointer to the object( obj ).
// The constructor initializes the propertyName and uses lambdas to adapt the member function pointers into std::function objects that conform to the expected
// getter and setter signatures.The getter lambda invokes the getter method on the object, and the setter lambda ensures the correct variant type is passed before
// invoking the setter method.
struct FeatureObjectSharedProperty
{
    std::string propertyName;
    FeaturePropertyKind kind;
    // due to getAllSharedProperties in FeatureObject returns static vector, we need externaly setup object to invoke setter ad getter.
    std::function<FeaturesPropertyTypesVariant( const FeatureObject* objectToInvoke, ViewportId id )> getter;
    // NOTE: `id` should usually be `{}`, not the current viewport ID, to set the property for all viewports.
    // Passing a non-zero ID would only modify the active viewport, and per-viewport properties aren't usually used.
    std::function<void( const FeaturesPropertyTypesVariant&, FeatureObject* objectToInvoke, ViewportId id )> setter;

    template <typename T, typename C, typename SetterFunc>
    FeatureObjectSharedProperty(
        std::string name,
        FeaturePropertyKind kind,
        T( C::* m_getter )( ViewportId ) const,
        SetterFunc m_setter
    )
        : propertyName( std::move( name ) ),
        kind( kind ),
        getter( [m_getter] ( const FeatureObject* objectToInvoke, ViewportId id ) -> FeaturesPropertyTypesVariant
        {
            return std::invoke( m_getter, dynamic_cast< const C* >( objectToInvoke ), id );
        } )
    {
        if constexpr ( ( std::is_same_v<SetterFunc, void ( C::* )( const T&, ViewportId )> )
            || ( std::is_same_v<SetterFunc, void ( C::* )( T, ViewportId )> ) )
        {
            setter = [m_setter] ( const FeaturesPropertyTypesVariant& v, FeatureObject* objectToInvoke, ViewportId id )
            {
                assert( std::holds_alternative<T>( v ) );
                if ( std::holds_alternative<T>( v ) )
                {
                    std::invoke( m_setter, dynamic_cast< C* > ( objectToInvoke ), std::get<T>( v ), id );
                }
            };
        }
        else
        {
            static_assert( dependent_false<T>, "Setter function signature unsupported" );
        }
    }
};

struct FeatureObjectProjectPointResult {
    Vector3f point;
    std::optional<Vector3f> normal;
};

enum class MRMESH_CLASS FeatureVisualizePropertyType
{
    Subfeatures,
    DetailsOnNameTag, // If true, show additional details on the name tag, such as point coordinates. Not all features use this.
    _count [[maybe_unused]],
};
template <> struct IsVisualizeMaskEnum<FeatureVisualizePropertyType> : std::true_type {};

/// An interface class which allows feature objects to share setters and getters on their main properties, for convenient presentation in the UI
class MRMESH_CLASS FeatureObject : public VisualObject
{
public:
    constexpr static const char* TypeName() noexcept { return "FeatureObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Feature"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Features"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// Create and generate list of bounded getters and setters for the main properties of feature object, together with prop. name for display and edit into UI.
    virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const = 0;

    [[nodiscard]] MRMESH_API bool supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const override;
    MRMESH_API AllVisualizeProperties getAllVisualizeProperties() const override;
    MRMESH_API const ViewportMask& getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;
    MRMESH_API void deserializeFields_( const Json::Value& root ) override;


    // Since a point on an abstract feature is difficult to uniquely parameterize,
    // the projection function simultaneously returns the normal to the surface at the projection point.
    [[nodiscard]] virtual FeatureObjectProjectPointResult projectPoint( const Vector3f& point, ViewportId id = {} ) const = 0;
    [[nodiscard]] MRMESH_API std::optional<Vector3f> getNormal( const Vector3f& point ) const;

    MRMESH_API void setXf( const AffineXf3f& xf, ViewportId id = {} ) override;
    MRMESH_API void resetXf( ViewportId id = {} ) override;

    // Returns point considered as base for the feature
    [[nodiscard]] MRMESH_API virtual Vector3f getBasePoint( ViewportId id = {} ) const;

    // The cached orthonormalized rotation matrix.
    // `isDef` receives false if matrix is overridden for this specific viewport.
    [[nodiscard]] Matrix3f getRotationMatrix( ViewportId id = {}, bool* isDef = nullptr ) const { return r_.get( id, isDef ); }
    // The cached scale and shear matrix. The main diagnoal stores the scale, and some other elements store the shearing.
    // `isDef` receives false if matrix is overridden for this specific viewport.
    [[nodiscard]] Matrix3f getScaleShearMatrix( ViewportId id = {}, bool* isDef = nullptr ) const { return s_.get( id, isDef ); }

    // This color is used for subfeatures.
    // `isDef` receives false if matrix is overridden for this specific viewport.
    MRMESH_API const Color& getDecorationsColor( bool selected, ViewportId viewportId = {}, bool* isDef = nullptr ) const;
    MRMESH_API virtual void setDecorationsColor( const Color& color, bool selected, ViewportId viewportId = {} );
    MRMESH_API virtual const ViewportProperty<Color>& getDecorationsColorForAllViewports( bool selected ) const;
    MRMESH_API virtual void setDecorationsColorForAllViewports( ViewportProperty<Color> val, bool selected );

    // Point size and line width, for primary rendering rather than subfeatures.
    [[nodiscard]] MRMESH_API virtual float getPointSize() const;
    [[nodiscard]] MRMESH_API virtual float getLineWidth() const;
    MRMESH_API virtual void setPointSize( float pointSize );
    MRMESH_API virtual void setLineWidth( float lineWidth );

    // Point size and line width, for subfeatures rather than primary rendering.
    [[nodiscard]] MRMESH_API virtual float getSubfeaturePointSize() const;
    [[nodiscard]] MRMESH_API virtual float getSubfeatureLineWidth() const;
    MRMESH_API virtual void setSubfeaturePointSize( float pointSize );
    MRMESH_API virtual void setSubfeatureLineWidth( float lineWidth );

    // Per-component alpha multipliers. The global alpha is multiplied by thise.
    [[nodiscard]] MRMESH_API virtual float getMainFeatureAlpha() const;
    [[nodiscard]] MRMESH_API virtual float getSubfeatureAlphaPoints() const;
    [[nodiscard]] MRMESH_API virtual float getSubfeatureAlphaLines() const;
    [[nodiscard]] MRMESH_API virtual float getSubfeatureAlphaMesh() const;
    MRMESH_API virtual void setMainFeatureAlpha( float alpha );
    MRMESH_API virtual void setSubfeatureAlphaPoints( float alpha );
    MRMESH_API virtual void setSubfeatureAlphaLines( float alpha );
    MRMESH_API virtual void setSubfeatureAlphaMesh( float alpha );

protected:
    // `numDimensions` is 0 for points, 1 for lines, 2 for surface meshes. We don't use 3 at the moment.
    MRMESH_API FeatureObject( int numDimensions );

    MRMESH_API void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos ) override;

    ViewportMask subfeatureVisibility_ = ViewportMask::all();
    ViewportMask detailsOnNameTag_ = ViewportMask::all();

    // Decomposition of the transformation matrix xf.A into a rotation and scaling matrix.Updated automatically in the setXf() method
    // This cache need for fast calculation of feature properties w/o expensive  transformation matrix QR decomposition.
    ViewportProperty<Matrix3f> r_; // rotation
    ViewportProperty<Matrix3f> s_; // scale

    // This is used for subfeatures. The index is for `isSelected()`.
    std::array<ViewportProperty<Color>, 2> decorationsColor_;

    // Those apply only to some features:

    // Point size and line width, for primary rendering rather than subfeatures.
    float pointSize_ = 10;
    float lineWidth_ = 2;

    // Point size and line width, for subfeatures rather than primary rendering.
    float subPointSize_ = 6;
    float subLineWidth_ = 1;

    // Per-component alpha multipliers. The global alpha is multiplied by thise.
    float mainFeatureAlpha_ = 1;
    float subAlphaPoints_ = 1;
    float subAlphaLines_ = 1;
    float subAlphaMesh_ = 0.5f;
};

}
