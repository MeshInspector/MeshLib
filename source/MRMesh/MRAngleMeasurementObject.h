#pragma once

#include "MRMesh/MRMeasurementObject.h"

namespace MR
{

// Represents an angle measurement.
class MRMESH_CLASS AngleMeasurementObject : public MeasurementObject
{
    // The xf encodes the two rays: the origin is the angle point, (1,0,0) is the first ray, (0,1,0) is the second ray.
public:
    AngleMeasurementObject() {}

    AngleMeasurementObject( AngleMeasurementObject&& ) noexcept = default;
    AngleMeasurementObject& operator=( AngleMeasurementObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "AngleMeasurementObject"; }
    const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Angle"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Angles"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    // For `std::make_shared()` in `clone()`.
    AngleMeasurementObject( ProtectedStruct, const AngleMeasurementObject& obj ) : AngleMeasurementObject( obj ) {}

    MRMESH_API std::shared_ptr<Object> clone() const override;
    MRMESH_API std::shared_ptr<Object> shallowClone() const override;

    // Get the angle point in world coordinates.
    [[nodiscard]] MRMESH_API Vector3f getWorldPoint() const;
    // Get the angle point in local coordinates.
    [[nodiscard]] MRMESH_API Vector3f getLocalPoint() const;

    // One of the two rays representing the angle, relative to the starting point.
    // They can have length != 1 for visualization purposes, it's probably a good idea to take the smaller of the two lengths.
    [[nodiscard]] MRMESH_API Vector3f getWorldRay( bool second ) const;
    // Same, but in local coordinates.
    [[nodiscard]] MRMESH_API Vector3f getLocalRay( bool second ) const;

    // Set the angle point in the local coordinates.
    MRMESH_API virtual void setLocalPoint( const MR::Vector3f& point );
    // Set the two rays representing the angle in the local coordinates.
    // The lengths are preserved.
    MRMESH_API virtual void setLocalRays( const MR::Vector3f& a, const MR::Vector3f& b );

    // Whether this is a conical angle. The middle line between the rays is preserved, but the rays themselves can be rotated.
    [[nodiscard]] MRMESH_API bool getIsConical() const;
    MRMESH_API void setIsConical( bool value );

    // Whether we should draw a ray from the center point to better visualize the angle. Enable this if there isn't already a line object there.
    [[nodiscard]] MRMESH_API bool getShouldVisualizeRay( bool second ) const;
    MRMESH_API void setShouldVisualizeRay( bool second, bool enable );

    // Computes the angle value, as if by `acos(dot(...))` from the two normalized `getWorldRay()`s.
    [[nodiscard]] MRMESH_API float computeAngle() const;

    [[nodiscard]] MRMESH_API std::vector<std::string> getInfoLines() const override;

protected:
    AngleMeasurementObject( const AngleMeasurementObject& other ) = default;

    MRMESH_API void swapBase_( Object& other ) override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;
    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API void setupRenderObject_() const override;

    MRMESH_API void onWorldXfChanged_() override;

private:
    // Don't forget to add all the new fields to serialization.

    // Whether this is a conical angle. The middle line between the rays is preserved, but the rays themselves can be rotated.
    bool isConical_ = false;

    // Whether we should draw a ray from the center point to better visualize the angle. Enable this if there isn't already a line object there.
    bool shouldVisualizeRay_[2]{};

    // The cached value for `computeAngle()`.
    mutable std::optional<float> cachedValue_;
};

} // namespace MR
