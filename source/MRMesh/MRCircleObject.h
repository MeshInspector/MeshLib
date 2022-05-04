#pragma once
#include "MRMeshFwd.h"
#include "MRObjectLinesHolder.h"

namespace MR
{
/// \defgroup FeaturesGroup Features
/// \ingroup DataModelGroup

/// Object to show sphere feature, position and radius are controlled by xf
/// \ingroup FeaturesGroup
class MRMESH_CLASS CircleObject : public ObjectLinesHolder
{
public:
    /// Creates simple sphere object with center in zero and radius - 1
    MRMESH_API CircleObject();
    /// Finds best sphere to approx given points
    MRMESH_API CircleObject( const std::vector<Vector3f>& pointsToApprox );

    CircleObject( CircleObject&& ) noexcept = default;
    CircleObject& operator = ( CircleObject&& ) noexcept = default;
    virtual ~CircleObject() = default;

    constexpr static const char* TypeName() noexcept { return "CircleObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    CircleObject( ProtectedStruct, const CircleObject& obj ) : CircleObject( obj )
    {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates radius from xf
    MRMESH_API float getRadius() const;
    /// calculates center from xf
    MRMESH_API Vector3f getCenter() const;
    /// calculates normal from xf
    MRMESH_API Vector3f getNormal() const;
    /// updates xf to fit given radius
    MRMESH_API void setRadius( float radius );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center );
    /// updates xf to fit given normal
    MRMESH_API void setNormal( const Vector3f& normal );
protected:
    CircleObject( const CircleObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& ) const override
    { return {}; }

    virtual tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& ) override
    { return {}; }

private:
    void constructPolyline_();
};

}