#pragma once

#include "MRFeatureObject.h"
#include "MRMesh/MRObjectComparableWithReference.h"
#include "MRMeshFwd.h"
#include "MRVisualObject.h"

namespace MR
{

/// Object to show point feature
/// \ingroup FeaturesGroup
class MRMESH_CLASS PointObject : public FeatureObject, public ObjectComparableWithReference
{
public:
    /// Creates simple point object with zero position
    MRMESH_API PointObject();
    /// Finds best point to approx given points
    MRMESH_API PointObject( const std::vector<Vector3f>& pointsToApprox );

    PointObject( PointObject&& ) noexcept = default;
    PointObject& operator = ( PointObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "PointObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Point"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Points"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    PointObject( ProtectedStruct, const PointObject& obj ) : PointObject( obj )
    {}

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates point from xf
    [[nodiscard]] MRMESH_API Vector3f getPoint( ViewportId id = {} ) const;
    /// updates xf to fit given point
    MRMESH_API void setPoint( const Vector3f& point, ViewportId id = {} );

    MRMESH_API virtual std::vector<FeatureObjectSharedProperty>& getAllSharedProperties() const override;

    [[nodiscard]] MRMESH_API FeatureObjectProjectPointResult projectPoint( const Vector3f& /*point*/, ViewportId id = {} ) const override;

    // Implement `ObjectComparableWithReference`:
    [[nodiscard]] MRMESH_API std::size_t numComparableProperties() const override;
    [[nodiscard]] MRMESH_API std::string_view getComparablePropertyName( std::size_t i ) const override;
    [[nodiscard]] MRMESH_API std::optional<ComparableProperty> computeComparableProperty( std::size_t i ) const override;
    [[nodiscard]] MRMESH_API std::optional<ComparisonTolerance> getComparisonTolerence( std::size_t i ) const override;
    MRMESH_API void setComparisonTolerance( std::size_t i, std::optional<ComparisonTolerance> newTolerance ) override;
    [[nodiscard]] MRMESH_API bool comparisonToleranceIsAlwaysOnlyPositive( std::size_t i ) const override;
    // This returns 2: the point, and the optional normal direction. The normal doesn't need to be normalized, its length doesn't affect calculations.
    // If the normal isn't specified, the euclidean distance gets used.
    [[nodiscard]] MRMESH_API std::size_t numComparisonReferenceValues() const override;
    [[nodiscard]] MRMESH_API std::string_view getComparisonReferenceValueName( std::size_t i ) const override;
    [[nodiscard]] MRMESH_API ComparisonReferenceValue getComparisonReferenceValue( std::size_t i ) const override;
    MRMESH_API void setComparisonReferenceValue( std::size_t i, std::optional<ComparisonReferenceValue::Var> value ) override;

protected:
    PointObject( const PointObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& ) const override
        { return {}; }

    virtual Expected<void> deserializeModel_( const std::filesystem::path&, ProgressCallback ) override
        { return {}; }

    MRMESH_API void setupRenderObject_() const override;

    // Quality control:
    std::optional<Vector3f> referencePos_;
    std::optional<Vector3f> referenceNormal_; // Not necessarily normalized.
    std::optional<ComparisonTolerance> tolerance_;
};

}
