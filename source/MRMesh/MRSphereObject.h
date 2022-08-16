#pragma once
#include "MRMeshFwd.h"
#include "MRObjectMeshHolder.h"

namespace MR
{

/// Object to show sphere feature, position and radius are controlled by xf
/// \ingroup FeaturesGroup
class MRMESH_CLASS SphereObject : public ObjectMeshHolder
{
public:
    /// Creates simple sphere object with center in zero and radius - 1
    MRMESH_API SphereObject();
    /// Finds best sphere to approx given points
    MRMESH_API SphereObject( const std::vector<Vector3f>& pointsToApprox );

    SphereObject( SphereObject&& ) noexcept = default;
    SphereObject& operator = ( SphereObject&& ) noexcept = default;
    virtual ~SphereObject() = default;

    constexpr static const char* TypeName() noexcept { return "SphereObject"; }
    virtual const char* typeName() const override {return TypeName(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    SphereObject( ProtectedStruct, const SphereObject& obj ) : SphereObject( obj ) {}

    virtual std::string getClassName() const override { return "Sphere"; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates radius from xf
    MRMESH_API float getRadius() const;
    /// calculates center from xf
    MRMESH_API Vector3f getCenter() const;
    /// updates xf to fit given radius
    MRMESH_API void setRadius( float radius );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center );
protected:
    SphereObject( const SphereObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& ) const override 
    { return {}; }

    virtual tl::expected<void, std::string> deserializeModel_( const std::filesystem::path&, ProgressCallback ) override
    { return {}; }

private:
    void constructMesh_();
};

}