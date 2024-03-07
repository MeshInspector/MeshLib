#pragma once

#include "MRMesh/MRObjectWithMeasurements.h"
#include "MRMeshFwd.h"
#include "MRFeatureObject.h"
#include "MRVisualObject.h"

namespace MR
{

/// Object to show Cone feature, position and radius are controlled by xf
/// \ingroup FeaturesGroup
class MRMESH_CLASS ConeObject : public ObjectWithMeasurements<FeatureObject, RadiusVisualizePropertyType(1), AngleVisualizePropertyType(1), LengthVisualizePropertyType(1)>
{
public:
    /// Creates simple Cone object with center in zero and radius - 1
    MRMESH_API ConeObject();
    /// Finds best Cone to approx given points
    MRMESH_API ConeObject( const std::vector<Vector3f>& pointsToApprox );

    ConeObject( ConeObject&& ) noexcept = default;
    ConeObject& operator = ( ConeObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept
    {
        return "ConeObject";
    }
    virtual const char* typeName() const override
    {
        return TypeName();
    }

    /// \note this ctor is public only for std::make_shared used inside clone()
    ConeObject( ProtectedStruct, const ConeObject& obj ) : ConeObject( obj )
    {}

    virtual std::string getClassName() const override
    {
        return "Cone";
    }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates cone angle from xf. It is an angle betweeh main axis and side.
    MRMESH_API float getAngle() const;
    /// calculates center from xf. Center is the apex of the cone.
    MRMESH_API Vector3f getCenter() const;
    /// calculates cone height from xf
    MRMESH_API float getHeight() const;
    /// calculates main axis direction from xf
    MRMESH_API Vector3f getDirection() const;
    /// updates xf to fit given center.  Center is the apex of the cone.
    MRMESH_API void setCenter( const Vector3f& center );
    /// updates xf to fit main axis
    MRMESH_API void setDirection( const Vector3f& normal );
    /// updates xf to fit cone height
    MRMESH_API void setHeight( float height );
    /// updates xf to fit given cone angle.  It is an angle betweeh main axis and side
    MRMESH_API void setAngle( float angle );
    /// Computes the base base radius from the xf.
    MRMESH_API float getBaseRadius() const;
    /// Updates the xf for the new base radius.
    MRMESH_API void setBaseRadius( float radius );

    MRMESH_API virtual const std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;


protected:
    ConeObject( const ConeObject& other ) = default;

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

    MeasurementPropertyParameters<RadiusVisualizePropertyType> getMeasurementParametersFor_( RadiusVisualizePropertyType index ) const override;
    MeasurementPropertyParameters<AngleVisualizePropertyType> getMeasurementParametersFor_( AngleVisualizePropertyType index ) const override;
    MeasurementPropertyParameters<LengthVisualizePropertyType> getMeasurementParametersFor_( LengthVisualizePropertyType index ) const override;

private:
    // Featue Radius fully controll by cone angle, but its need for speedup internal calculation (not use tan / atan from each estimation).
    float getNormalizedRadius_() const;
};

}
