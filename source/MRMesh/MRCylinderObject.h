#pragma once

#include "MRFeatureObject.h"
#include "MRMeshFwd.h"
#include "MRVisualObject.h"

namespace MR
{

/// Object to show Cylinder feature, position and radius are controlled by xf
/// \ingroup FeaturesGroup
class MRMESH_CLASS CylinderObject : public VisualObject, public FeatureObject
{
public:
    /// Creates simple Cylinder object with center in zero and radius - 1
    MRMESH_API CylinderObject();
    /// Finds best Cylinder to approx given points
    MRMESH_API CylinderObject( const std::vector<Vector3f>& pointsToApprox );

    CylinderObject( CylinderObject&& ) noexcept = default;
    CylinderObject& operator = ( CylinderObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept
    {
        return "CylinderObject";
    }
    virtual const char* typeName() const override
    {
        return TypeName();
    }

    /// \note this ctor is public only for std::make_shared used inside clone()
    CylinderObject( ProtectedStruct, const CylinderObject& obj ) : CylinderObject( obj )
    {}

    virtual std::string getClassName() const override
    {
        return "Cylinder";
    }

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
    /// calculates main axis direction from xf
    MRMESH_API Vector3f getDirection() const;
    /// updates xf to fit main axis
    MRMESH_API void setDirection( const Vector3f& normal );
    /// calculates cylinder length from xf
    MRMESH_API float getLength() const;
    /// updates xf to fit cylinder length
    MRMESH_API void setLength( float length );

    MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;


protected:
    CylinderObject( const CylinderObject& other ) = default;

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
