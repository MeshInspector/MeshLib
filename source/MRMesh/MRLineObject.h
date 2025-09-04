#pragma once
#include "MRPch/MRBindingMacros.h"
#include "MRMeshFwd.h"
#include "MRFeatureObject.h"
#include "MRVisualObject.h"

namespace MR
{

/// Object to show plane feature
/// \ingroup FeaturesGroup
class MRMESH_CLASS LineObject : public FeatureObject
{
public:
    /// Creates simple plane object
    MRMESH_API LineObject();
    /// Finds best plane to approx given points
    MRMESH_API LineObject( const std::vector<Vector3f>& pointsToApprox );

    LineObject( LineObject&& ) noexcept = default;
    LineObject& operator = ( LineObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "LineObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Line"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Lines"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    LineObject( ProtectedStruct, const LineObject& obj ) : LineObject( obj )
    {}

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates direction from xf
    MRMESH_API Vector3f getDirection( ViewportId id = {} ) const;
    /// calculates center from xf
    MRMESH_API Vector3f getCenter( ViewportId id = {} ) const;
    /// updates xf to fit given normal
    MRMESH_API void setDirection( const Vector3f& normal, ViewportId id = {} );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center, ViewportId id = {} );
    /// updates xf to scale size
    MRMESH_API void setLength( float size, ViewportId id = {} );
    /// calculates line size from xf
    [[nodiscard]] MRMESH_API float getLength( ViewportId id = {} ) const;
    // Returns point considered as base for the feature
    [[nodiscard]] MRMESH_API virtual Vector3f getBasePoint( ViewportId id = {} ) const override;

    /// Returns the starting point, aka `center - dir * len/2`.
    [[nodiscard]] MRMESH_API Vector3f getPointA( ViewportId id = {} ) const;
    /// Returns the finishing point, aka `center + dir * len/2`.
    [[nodiscard]] MRMESH_API Vector3f getPointB( ViewportId id = {} ) const;

    [[deprecated( "This confusingly sets half-length. Use `setLength(halfLen * 2)` instead." )]]
    MR_BIND_IGNORE void setSize( float halfLen, ViewportId id = {} )
    {
        setLength( halfLen * 2 , id );
    }

    [[nodiscard]] MRMESH_API FeatureObjectProjectPointResult projectPoint( const Vector3f& point, ViewportId id = {} ) const override;

    MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

protected:
    LineObject( const LineObject& other ) = default;

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
