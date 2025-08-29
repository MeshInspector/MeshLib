#pragma once

#include "MRFeatureObject.h"
#include "MRMeshFwd.h"
#include "MRVisualObject.h"

namespace MR
{

/// Object to show point feature
/// \ingroup FeaturesGroup
class MRMESH_CLASS PointObject : public FeatureObject
{
public:
    /// Creates simple point object with zero position
    MRMESH_API PointObject();
    /// Finds best point to approx given points
    MRMESH_API PointObject( const std::vector<Vector3f>& pointsToApprox );

    PointObject( PointObject&& ) noexcept = default;
    PointObject& operator = ( PointObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "PointObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Point"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Points"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    PointObject( ProtectedStruct, const PointObject& obj ) : PointObject( obj )
    {}

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates point from xf
    [[nodiscard]] MRMESH_API Vector3f getPoint( ViewportId id = {} ) const;
    /// updates xf to fit given point
    MRMESH_API void setPoint( const Vector3f& point, ViewportId id = {} );

    MRMESH_API virtual  std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

    [[nodiscard]] MRMESH_API FeatureObjectProjectPointResult projectPoint( const Vector3f& /*point*/, ViewportId id = {} ) const override;

protected:
    PointObject( const PointObject& other ) = default;

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
