#pragma once

#include "MRMesh/MRMeasurementObject.h"

namespace MR
{

// Represents a radius measurement.
class MRMESH_CLASS RadiusMeasurementObject : public MeasurementObject
{
    // The xf encodes the radius: the origin is the center point, and (1,0,0) is the end point.
    // For non-spherical radiuses, (0,0,1) is the circle normal.
public:
    RadiusMeasurementObject() {}

    RadiusMeasurementObject( RadiusMeasurementObject&& ) noexcept = default;
    RadiusMeasurementObject& operator=( RadiusMeasurementObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "RadiusMeasurementObject"; }
    const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Radius"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Radii"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    // For `std::make_shared()` in `clone()`.
    RadiusMeasurementObject( ProtectedStruct, const RadiusMeasurementObject& obj ) : RadiusMeasurementObject( obj ) {}

    MRMESH_API std::shared_ptr<Object> clone() const override;
    MRMESH_API std::shared_ptr<Object> shallowClone() const override;

    // Get the center in world coordinates.
    [[nodiscard]] MRMESH_API Vector3f getWorldCenter() const;
    // Get the center in local coordinates.
    [[nodiscard]] MRMESH_API Vector3f getLocalCenter() const;

    // The length of this vector is the radius, and the direction is the preferred line drawing direction.
    [[nodiscard]] MRMESH_API Vector3f getWorldRadiusAsVector() const;
    [[nodiscard]] MRMESH_API Vector3f getLocalRadiusAsVector() const;

    // The preferred radius normal, for non-spherical radiuses.
    [[nodiscard]] MRMESH_API Vector3f getWorldNormal() const;
    [[nodiscard]] MRMESH_API Vector3f getLocalNormal() const;

    MRMESH_API virtual void setLocalCenter( const MR::Vector3f& center );
    // Sets the local radius vector (the length of which is the radius value),
    //   and also the radius normal (which is ignored for spherical radiuses).
    // The normal is automatically normalized and made perpendicular to the `radiusVec`.
    MRMESH_API virtual void setLocalRadiusAsVector( const MR::Vector3f& radiusVec, const Vector3f& normal );
    // Same, but without a preferred normal.
    void setLocalRadiusAsVector( const MR::Vector3f& radiusVec ) { setLocalRadiusAsVector( radiusVec, radiusVec.furthestBasisVector() ); }

    // Whether we should draw this as a diameter instead of a radius.
    [[nodiscard]] bool getDrawAsDiameter() const { return drawAsDiameter_; }
    MRMESH_API virtual void setDrawAsDiameter( bool value );

    // Whether this is a sphere radius, as opposed to circle/cylinder radius.
    [[nodiscard]] bool getIsSpherical() const { return isSpherical_; }
    virtual void setIsSpherical( bool value ) { isSpherical_ = value; }

    // The visual leader line length multiplier, relative to the radius.
    // You're recommended to set a min absolute value for the resulting length when rendering.
    [[nodiscard]] float getVisualLengthMultiplier() const { return visualLengthMultiplier_; }
    virtual void setVisualLengthMultiplier( float value ) { visualLengthMultiplier_ = value; }

    // Computes the radius/diameter value, as if by `getLocalRadiusAsVector()`, possibly multiplied by two if `getDrawAsDiameter()`.
    [[nodiscard]] MRMESH_API float computeRadiusOrDiameter() const;

    [[nodiscard]] MRMESH_API std::vector<std::string> getInfoLines() const override;

protected:
    RadiusMeasurementObject( const RadiusMeasurementObject& other ) = default;

    MRMESH_API void swapBase_( Object& other ) override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;
    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API void setupRenderObject_() const override;

    MRMESH_API void onWorldXfChanged_() override;

private:
    // Don't forget to add all the new fields to serialization.

    // Whether we should draw this object as a diameter instead of a radius.
    bool drawAsDiameter_ = false;
    // Whether this is a sphere radius, as opposed to circle/cylinder radius.
    bool isSpherical_ = false;

    // The visual leader line length multiplier, relative to the radius.
    // You're recommended to set a min absolute value for the resulting length when rendering.
    float visualLengthMultiplier_ = 2 / 3.f;

    // The cached value for `computeRadiusOrDiameter()`.
    mutable std::optional<float> cachedValue_;
};

} // namespace MR
