#pragma once

#include "MRMesh/MRAddVisualPropertiesMixin.h"
#include "MRMesh/MRObjectDimensionsEnum.h"
#include "MRMeshFwd.h"
#include "MRFeatureObject.h"
#include "MRVisualObject.h"

namespace MR
{

/// Object to show Cone feature, position and radius are controlled by xf
/// \ingroup FeaturesGroup
class MRMESH_CLASS ConeObject : public AddVisualProperties<FeatureObject,
    DimensionsVisualizePropertyType::diameter,
    DimensionsVisualizePropertyType::angle,
    DimensionsVisualizePropertyType::length
>
{
public:
    /// Creates simple Cone object with center in zero and radius - 1
    MRMESH_API ConeObject();
    /// Finds best Cone to approx given points
    MRMESH_API ConeObject( const std::vector<Vector3f>& pointsToApprox );

    ConeObject( ConeObject&& ) noexcept = default;
    ConeObject& operator = ( ConeObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "ConeObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Cone"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Cones"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    ConeObject( ProtectedStruct, const ConeObject& obj ) : ConeObject( obj )
    {}

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates cone angle from xf. It is an angle betweeh main axis and side.
    [[nodiscard]] MRMESH_API float getAngle( ViewportId id = {} ) const;
    /// calculates center from xf. Center is the apex of the cone.
    [[nodiscard]] MRMESH_API Vector3f getCenter( ViewportId id = {} ) const;
    /// calculates cone height from xf
    [[nodiscard]] MRMESH_API float getHeight( ViewportId id = {} ) const;
    /// calculates main axis direction from xf
    [[nodiscard]] MRMESH_API Vector3f getDirection( ViewportId id = {} ) const;
    /// updates xf to fit given center.  Center is the apex of the cone.
    MRMESH_API void setCenter( const Vector3f& center, ViewportId id = {} );
    /// updates xf to fit main axis
    MRMESH_API void setDirection( const Vector3f& normal, ViewportId id = {} );
    /// updates xf to fit cone height
    MRMESH_API void setHeight( float height, ViewportId id = {} );
    /// updates xf to fit given cone angle.  It is an angle betweeh main axis and side
    MRMESH_API void setAngle( float angle, ViewportId id = {} );
    /// Computes the base base radius from the xf.
    [[nodiscard]] MRMESH_API float getBaseRadius( ViewportId id = {} ) const;
    /// Updates the xf for the new base radius.
    MRMESH_API void setBaseRadius( float radius, ViewportId id = {} );

    // Returns point considered as base for the feature
    [[nodiscard]] MRMESH_API virtual Vector3f getBasePoint( ViewportId id = {} ) const override;

    MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

    [[nodiscard]] MRMESH_API FeatureObjectProjectPointResult projectPoint( const Vector3f& point, ViewportId id = {} ) const override;

protected:
    ConeObject( const ConeObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& ) const override
    {
        return {};
    }

    virtual Expected<void> deserializeModel_( const std::filesystem::path&, ProgressCallback ) override
    {
        return {};
    }

    MRMESH_API void setupRenderObject_() const override;

private:
    // Featue Radius fully controll by cone angle, but its need for speedup internal calculation (not use tan / atan from each estimation).
    float getNormalizedRadius_( ViewportId id = {} ) const;
};

}
