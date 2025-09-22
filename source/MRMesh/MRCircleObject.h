#pragma once
#include "MRMesh/MRAddVisualPropertiesMixin.h"
#include "MRMesh/MRObjectDimensionsEnum.h"
#include "MRMeshFwd.h"
#include "MRFeatureObject.h"
#include "MRVisualObject.h"
#include "MRPlane3.h"

namespace MR
{
/// \defgroup FeaturesGroup Features
/// \ingroup DataModelGroup

/// Object to show sphere feature, position and radius are controlled by xf
/// \ingroup FeaturesGroup
class MRMESH_CLASS CircleObject : public AddVisualProperties<FeatureObject, DimensionsVisualizePropertyType::diameter>
{
public:
    /// Creates simple sphere object with center in zero and radius - 1
    MRMESH_API CircleObject();
    /// Finds best sphere to approx given points
    MRMESH_API CircleObject( const std::vector<Vector3f>& pointsToApprox );

    CircleObject( CircleObject&& ) noexcept = default;
    CircleObject& operator = ( CircleObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "CircleObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Circle"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Circles"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    CircleObject( ProtectedStruct, const CircleObject& obj ) : CircleObject( obj )
    {}

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates radius from xf
    [[nodiscard]] MRMESH_API float getRadius( ViewportId id = {} ) const;
    /// calculates center from xf
    [[nodiscard]] MRMESH_API  Vector3f getCenter( ViewportId id = {} ) const;
    /// calculates normal from xf
    [[nodiscard]] MRMESH_API Vector3f getNormal( ViewportId id = {} ) const;
    /// updates xf to fit given radius
    MRMESH_API void setRadius( float radius, ViewportId id = {} );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center, ViewportId id = {} );
    /// updates xf to fit given normal
    MRMESH_API void setNormal( const Vector3f& normal, ViewportId id = {} );

    [[nodiscard]] MRMESH_API FeatureObjectProjectPointResult projectPoint( const Vector3f& point, ViewportId id = {} ) const override;

    [[nodiscard]] MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

protected:
    CircleObject( const CircleObject& other ) = default;

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
};

}
