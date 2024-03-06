#pragma once
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
class MRMESH_CLASS CircleObject : public FeatureObject
{
public:
    /// Creates simple sphere object with center in zero and radius - 1
    MRMESH_API CircleObject();
    /// Finds best sphere to approx given points
    MRMESH_API CircleObject( const std::vector<Vector3f>& pointsToApprox );

    CircleObject( CircleObject&& ) noexcept = default;
    CircleObject& operator = ( CircleObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept
    {
        return "CircleObject";
    }
    virtual const char* typeName() const override
    {
        return TypeName();
    }

    /// \note this ctor is public only for std::make_shared used inside clone()
    CircleObject( ProtectedStruct, const CircleObject& obj ) : CircleObject( obj )
    {}

    virtual std::string getClassName() const override
    {
        return "Circle";
    }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates radius from xf
    [[nodiscard]] MRMESH_API float getRadius() const;
    /// calculates center from xf
    [[nodiscard]] MRMESH_API Vector3f getCenter() const;
    /// calculates normal from xf
    [[nodiscard]] MRMESH_API Vector3f getNormal() const;
    /// updates xf to fit given radius
    MRMESH_API void setRadius( float radius );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center );
    /// updates xf to fit given normal
    MRMESH_API void setNormal( const Vector3f& normal );

    [[nodiscard]] FeatureObjectProjectPointResult projectPoint( const Vector3f& point ) const override
    {
        const Vector3f center = getCenter();
        const float radius = getRadius();
        auto normal = getNormal();

        Plane3f plane( normal, dot( normal, center ) );
        auto K = plane.project( point );
        auto n = ( K - center ).normalized();
        auto projection = center + n * radius;

        return { projection, std::nullopt };
    };

    [[nodiscard]] MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

protected:
    CircleObject( const CircleObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual Expected<std::future<VoidOrErrStr>> serializeModel_( const std::filesystem::path& ) const override
    {
        return {};
    }

    virtual VoidOrErrStr deserializeModel_( const std::filesystem::path&, ProgressCallback ) override
    {
        return {};
    }

    MRMESH_API void setupRenderObject_() const override;
};

}
