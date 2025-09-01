#pragma once

#include "MRMeshFwd.h"
#include "MRFeatureObject.h"

namespace MR
{

/// Object to show plane feature
/// \ingroup FeaturesGroup
class MRMESH_CLASS PlaneObject : public FeatureObject
{
public:
    /// Creates simple plane object
    MRMESH_API PlaneObject();
    /// Finds best plane to approx given points
    MRMESH_API PlaneObject( const std::vector<Vector3f>& pointsToApprox );

    PlaneObject( PlaneObject&& ) noexcept = default;
    PlaneObject& operator = ( PlaneObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "PlaneObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Plane"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Planes"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    PlaneObject( ProtectedStruct, const PlaneObject& obj ) : PlaneObject( obj )
    {}

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates normal from xf
    [[nodiscard]] MRMESH_API Vector3f getNormal( ViewportId id = {} ) const;
    /// calculates center from xf
    [[nodiscard]] MRMESH_API Vector3f getCenter( ViewportId id = {} ) const;
    /// updates xf to fit given normal
    MRMESH_API void setNormal( const Vector3f& normal, ViewportId id = {} );
    /// updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center, ViewportId id = {} );
    /// updates xf to scale size
    MRMESH_API void setSize( float size, ViewportId id = {} );
    /// calculates plane size from xf
    [[nodiscard]] MRMESH_API float getSize( ViewportId id = {} ) const;

    [[nodiscard]] MRMESH_API float getSizeX( ViewportId id = {} ) const;
    [[nodiscard]] MRMESH_API float getSizeY( ViewportId id = {} ) const;
    /// calculates normalized directions of X,Y axis of the plane and normal as Z
    [[nodiscard]] MRMESH_API Matrix3f calcLocalBasis( ViewportId id = {} ) const;

    MRMESH_API void setSizeX( float size, ViewportId id = {} );
    MRMESH_API void setSizeY( float size, ViewportId id = {} );

    // Returns point considered as base for the feature
    [[nodiscard]] MRMESH_API virtual Vector3f getBasePoint( ViewportId id = {} ) const override;

    [[nodiscard]] MRMESH_API FeatureObjectProjectPointResult projectPoint( const Vector3f& point, ViewportId id = {} ) const override;

    MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

protected:
    PlaneObject( const PlaneObject& other ) = default;

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
    void orientateFollowMainAxis_( ViewportId id = {} );
    void setupPlaneSize2DByOriginalPoints_( const std::vector<Vector3f>& pointsToApprox );
};

}
