#pragma once

#include "MRMeasurementObject.h"
#include "MRObjectComparableWithReference.h"

namespace MR
{

class MRMESH_CLASS PointMeasurementObject
    : public MeasurementObject
    , public ObjectComparableWithReference
{
public:
     PointMeasurementObject() = default;

    PointMeasurementObject( PointMeasurementObject&& ) noexcept = default;
    PointMeasurementObject& operator =( PointMeasurementObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "PointMeasurementObject"; }
    const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Measure Point"; }
    std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Measure Points"; }
    std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    PointMeasurementObject( ProtectedStruct, const PointMeasurementObject& obj ) : PointMeasurementObject( obj )
    {}

    MRMESH_API std::shared_ptr<Object> clone() const override;
    MRMESH_API std::shared_ptr<Object> shallowClone() const override;

    /// calculates point from xf
    [[nodiscard]] MRMESH_API Vector3f getPoint( ViewportId id = {} ) const;
    /// updates xf to fit given point
    MRMESH_API void setPoint( const Vector3f& point, ViewportId id = {} );

    // Implement `ObjectComparableWithReference`:
    [[nodiscard]] MRMESH_API std::size_t numComparableProperties() const override;
    [[nodiscard]] MRMESH_API std::string_view getComparablePropertyName( std::size_t i ) const override;
    [[nodiscard]] MRMESH_API std::optional<ComparableProperty> computeComparableProperty( std::size_t i ) const override;
    [[nodiscard]] MRMESH_API std::optional<ComparisonTolerance> getComparisonTolerence( std::size_t i ) const override;
    MRMESH_API void setComparisonTolerance( std::size_t i, std::optional<ComparisonTolerance> newTolerance ) override;
    [[nodiscard]] MRMESH_API bool comparisonToleranceIsAlwaysOnlyPositive( std::size_t i ) const override;
    // This returns 2: the point, and the optional normal direction. The normal doesn't need to be normalized, its length doesn't affect calculations.
    // If the normal isn't specified, the Euclidean distance gets used.
    [[nodiscard]] MRMESH_API std::size_t numComparisonReferenceValues() const override;
    [[nodiscard]] MRMESH_API std::string_view getComparisonReferenceValueName( std::size_t i ) const override;
    [[nodiscard]] MRMESH_API ComparisonReferenceValue getComparisonReferenceValue( std::size_t i ) const override;
    MRMESH_API void setComparisonReferenceValue( std::size_t i, std::optional<ComparisonReferenceValue::Var> value ) override;

protected:
    PointMeasurementObject( const PointMeasurementObject& other ) = default;

    /// swaps this object with other
    MRMESH_API void swapBase_( Object& other ) override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;
    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API void setupRenderObject_() const override;

 private:
    std::optional<Vector3f> referencePos_;
    std::optional<Vector3f> referenceNormal_; // Not necessarily normalized.
    std::optional<ComparisonTolerance> tolerance_;
};

} // namespace MR
