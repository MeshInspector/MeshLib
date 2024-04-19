#pragma once

#include "MRMesh/MRMeasurementObject.h"

namespace MR
{

// Represents a distance measurement.
class MRMESH_CLASS DistanceMeasurementObject : public MeasurementObject
{
    // The xf encodes the distance: the origin is one point, and (1,0,0) is another.
public:
    DistanceMeasurementObject() {}

    DistanceMeasurementObject( DistanceMeasurementObject&& ) noexcept = default;
    DistanceMeasurementObject& operator=( DistanceMeasurementObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "DistanceMeasurementObject"; }
    const char* typeName() const override { return TypeName(); }

    // For `std::make_shared()` in `clone()`.
    DistanceMeasurementObject( ProtectedStruct, const DistanceMeasurementObject& obj ) : DistanceMeasurementObject( obj ) {}

    std::string getClassName() const override { return "Distance"; }

    MRMESH_API std::shared_ptr<Object> clone() const override;
    MRMESH_API std::shared_ptr<Object> shallowClone() const override;

    // Get the starting point in world coordinates.
    [[nodiscard]] MRMESH_API Vector3f getWorldPoint() const;
    // Get the starting point in local coordinates.
    [[nodiscard]] MRMESH_API Vector3f getLocalPoint() const;

    // The delta from the starting point to the other point.
    [[nodiscard]] MRMESH_API Vector3f getWorldDelta() const;
    [[nodiscard]] MRMESH_API Vector3f getLocalDelta() const;

    // Set the start point in the local coordinates.
    MRMESH_API virtual void setLocalPoint( const MR::Vector3f& point );
    // Set the delta vector in the local coordinates.
    MRMESH_API virtual void setLocalDelta( const MR::Vector3f& delta );

    // Whether the distance should be displayed as a negative one.
    [[nodiscard]] MRMESH_API bool getDrawAsNegative() const;
    MRMESH_API virtual void setDrawAsNegative( bool value );

    enum class PerCoordDeltas
    {
        none, // Hide.
        withSign, // Display as is.
        absolute, // Display absolute values.
    };
    // Whether we should draw the individual X/Y/Z deltas in addition to the distance itself.
    [[nodiscard]] MRMESH_API PerCoordDeltas getPerCoordDeltasMode() const;
    MRMESH_API virtual void setPerCoordDeltasMode( PerCoordDeltas mode );

    // Computes the distance value: `getWorldDelta().length() * (getDrawAsNegative() ? -1 : 1)`.
    [[nodiscard]] MRMESH_API float computeDistance() const;

    [[nodiscard]] MRMESH_API std::vector<std::string> getInfoLines() const override;

protected:
    DistanceMeasurementObject( const DistanceMeasurementObject& other ) = default;

    MRMESH_API void swapBase_( Object& other ) override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;
    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API void setupRenderObject_() const override;

    MRMESH_API void propagateWorldXfChangedSignal_() override;

private:
    // Don't forget to add all the new fields to serialization.

    // Whether the distance should be displayed as a negative one.
    bool drawAsNegative_ = false;

    // Whether we should draw the individual X/Y/Z deltas in addition to the distance itself.
    PerCoordDeltas perCoordDeltas_ = PerCoordDeltas::none;

    // The cached value for `computeDistance()`.
    mutable std::optional<float> cachedValue_;
};

} // namespace MR
