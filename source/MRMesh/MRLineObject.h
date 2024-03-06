#pragma once
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

    /// \note this ctor is public only for std::make_shared used inside clone()
    LineObject( ProtectedStruct, const LineObject& obj ) : LineObject( obj )
    {}

    virtual std::string getClassName() const override { return "Line"; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates direction from xf
    MRMESH_API Vector3f getDirection() const;
    /// calculates center from xf
    MRMESH_API Vector3f getCenter() const;
    /// updates xf to fit given normal
    MRMESH_API void setDirection( const Vector3f& normal );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center );
    /// updates xf to scale size
    MRMESH_API void setLength( float size );
    /// calculates line size from xf
    [[nodiscard]] MRMESH_API float getLength() const;

    /// Returns the starting point, aka `center - dir * len/2`.
    [[nodiscard]] MRMESH_API Vector3f getPointA() const;
    /// Returns the finishing point, aka `center + dir * len/2`.
    [[nodiscard]] MRMESH_API Vector3f getPointB() const;

    [[deprecated("This confusingly sets half-length. Use `setLength(halfLen * 2)` instead.")]]
    void setSize( float halfLen ) { setLength( halfLen * 2 ); }

    [[nodiscard]] FeatureObjectProjectPointResult projectPoint( const Vector3f& point ) const override
    {
        const Vector3f center = getCenter();
        const Vector3f direction = getDirection();

        auto X = point - center;
        auto K = direction * dot( X, direction );

        return { K + center , {} };
    };

    MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

protected:
    LineObject( const LineObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual Expected<std::future<VoidOrErrStr>> serializeModel_( const std::filesystem::path& ) const override
        { return {}; }

    virtual VoidOrErrStr deserializeModel_( const std::filesystem::path&, ProgressCallback ) override
        { return {}; }

    MRMESH_API void setupRenderObject_() const override;
};

}
