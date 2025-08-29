#pragma once

#include "MRFeatureObject.h"
#include "MRMesh/MRAddVisualPropertiesMixin.h"
#include "MRMesh/MRObjectDimensionsEnum.h"
#include "MRMeshFwd.h"
#include "MRVisualObject.h"

namespace MR
{

/// Object to show sphere feature, position and radius are controlled by xf
/// \ingroup FeaturesGroup
class MRMESH_CLASS SphereObject : public AddVisualProperties<FeatureObject, DimensionsVisualizePropertyType::diameter>
{
public:
    /// Creates simple sphere object with center in zero and radius - 1
    MRMESH_API SphereObject();
    /// Finds best sphere to approx given points
    MRMESH_API SphereObject( const std::vector<Vector3f>& pointsToApprox );

    SphereObject( SphereObject&& ) noexcept = default;
    SphereObject& operator = ( SphereObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "SphereObject"; }
    virtual const char* typeName() const override {return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Sphere"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Spheres"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    SphereObject( ProtectedStruct, const SphereObject& obj ) : SphereObject( obj ) {}

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates radius from xf
    MRMESH_API float getRadius( ViewportId id = {} ) const;
    /// calculates center from xf
    MRMESH_API Vector3f getCenter( ViewportId id = {} ) const;
    /// updates xf to fit given radius
    MRMESH_API void setRadius( float radius, ViewportId id = {} );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center, ViewportId id = {} );

    MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

    [[nodiscard]] MRMESH_API FeatureObjectProjectPointResult projectPoint( const Vector3f& point, ViewportId id = {} ) const override;

protected:
    SphereObject( const SphereObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& ) const override
        { return {}; }

    virtual Expected<void> deserializeModel_( const std::filesystem::path&, ProgressCallback ) override
        { return {}; }

    MRMESH_API void setupRenderObject_() const override;
};

}
