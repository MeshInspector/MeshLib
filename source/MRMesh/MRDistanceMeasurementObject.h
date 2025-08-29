#pragma once

#include "MRMesh/MRMeasurementObject.h"
#include "MRMesh/MRObjectComparableWithReference.h"

namespace MR
{

// Represents a distance measurement.
class MRMESH_CLASS DistanceMeasurementObject : public MeasurementObject, public ObjectComparableWithReference
{
    // The xf encodes the distance: the origin is one point, and (1,0,0) is another.
public:
    DistanceMeasurementObject() {}

    DistanceMeasurementObject( DistanceMeasurementObject&& ) noexcept = default;
    DistanceMeasurementObject& operator=( DistanceMeasurementObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "DistanceMeasurementObject"; }
    const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Distance"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Distances"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    // For `std::make_shared()` in `clone()`.
    DistanceMeasurementObject( ProtectedStruct, const DistanceMeasurementObject& obj ) : DistanceMeasurementObject( obj ) {}

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
    [[nodiscard]] MRMESH_API bool isNegative() const;
    MRMESH_API virtual void setIsNegative( bool value );

    enum class DistanceMode
    {
        eucledian, // Eucledian distance.
        eucledianWithSignedDeltasPerAxis, // Eucledian distance, but also display per-axis deltas with signs.
        eucledianWithAbsoluteDeltasPerAxis, // Eucledian distance, but also display per-axis deltas without signs.
        // Absolute distance in one axis:
        xAbsolute,
        yAbsolute,
        zAbsolute,
    };
    // Whether we should draw the individual X/Y/Z deltas in addition to the distance itself.
    [[nodiscard]] MRMESH_API DistanceMode getDistanceMode() const;
    MRMESH_API virtual void setDistanceMode( DistanceMode mode );

    // Computes the distance value. This is affected by `getDistanceMode()`.
    // In `eucledian`, this is `getWorldDelta().length() * (isNegative() ? -1 : 1)`.
    [[nodiscard]] MRMESH_API float computeDistance() const;

    [[nodiscard]] MRMESH_API std::vector<std::string> getInfoLines() const override;

    // Implement `ObjectComparableWithReference`:
    [[nodiscard]] MRMESH_API std::size_t numComparableProperties() const override;
    [[nodiscard]] MRMESH_API std::string_view getComparablePropertyName( std::size_t i ) const override;
    [[nodiscard]] MRMESH_API std::optional<float> compareProperty( const Object& other, std::size_t i ) const override;
    [[nodiscard]] MRMESH_API bool hasComparisonTolerances() const override;
    [[nodiscard]] MRMESH_API ComparisonTolerance getComparisonTolerences( std::size_t i ) const override;
    MRMESH_API void setComparisonTolerance( std::size_t i, const ComparisonTolerance& newTolerance ) override;
    MRMESH_API void resetComparisonTolerances() override;
    [[nodiscard]] MRMESH_API bool hasComparisonReferenceValues() const override;
    [[nodiscard]] MRMESH_API float getComparisonReferenceValue( std::size_t i ) const override;
    MRMESH_API void setComparisonReferenceValue( std::size_t i, float value ) override;
    MRMESH_API void resetComparisonReferenceValues() override;

protected:
    DistanceMeasurementObject( const DistanceMeasurementObject& other ) = default;

    MRMESH_API void swapBase_( Object& other ) override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;
    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API void setupRenderObject_() const override;

    MRMESH_API void onWorldXfChanged_() override;

private:
    // Don't forget to add all the new fields to serialization.

    // Whether the distance should be displayed as a negative one.
    bool isNegative_ = false;

    // Whether we should draw the individual X/Y/Z deltas in addition to the distance itself.
    DistanceMode perCoordDeltas_ = DistanceMode::eucledian;

    // The cached value for `computeDistance()`.
    mutable std::optional<float> cachedValue_;

    std::optional<ComparisonTolerance> tolerance_;

    std::optional<float> referenceValue_;
};

} // namespace MR
