#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"

#include <optional>
#include <variant>

namespace MR
{

// A base class for a data-model object that is a feature/measurement that can be compared between two models.
class MRMESH_CLASS ObjectComparableWithReference
{
  public:
    virtual ~ObjectComparableWithReference() = default;

    // We have no member variables, so lack of the implicit copy operations wouldn't matter, but MSVC warns on this, so we have to be explicit.
    ObjectComparableWithReference() = default;
    ObjectComparableWithReference( const ObjectComparableWithReference& ) = default;
    ObjectComparableWithReference( ObjectComparableWithReference&& ) = default;
    ObjectComparableWithReference& operator=( const ObjectComparableWithReference& ) = default;
    ObjectComparableWithReference& operator=( ObjectComparableWithReference&& ) = default;


    // Comparing properties:

    // When comparing this object with a reference, how many different properties can we output?
    [[nodiscard]] virtual std::size_t numComparableProperties() const = 0;

    // `i` goes up to `numComparableProperties()`, exclusive.
    [[nodiscard]] virtual std::string_view getComparablePropertyName( std::size_t i ) const = 0;

    struct ComparableProperty
    {
        float value = 0;

        // This can be null if the reference value isn't set, or something else is wrong.
        // This can match whatever is set via `get/setComparisonReferenceValue()`, but not necessarily.
        // E.g. for point coordinates, those functions act on the reference coordinates (three optional floats), but this number is always zero,
        //   and the `value` is the distance to those coordinates.
        std::optional<float> referenceValue = 0.f;
    };

    // Compute a value of a property.
    // Compare `value` and `referenceValue` using the tolerance.
    // This can return null if the value is impossible to compute, e.g. for some types if the reference isn't set (e.g. if
    //   we're computing the distance to a reference point).
    // `i` goes up to `numComparableProperties()`, exclusive.
    [[nodiscard]] virtual std::optional<ComparableProperty> computeComparableProperty( std::size_t i ) const = 0;


    // Tolerances:

    struct ComparisonTolerance
    {
        // How much larger can this value be compared to the reference?
        float positive = 0;
        // How much smaller can this value be compared to the reference?
        // This number should normally be zero or negative.
        float negative = 0;
    };

    // Returns the tolerance for a specific comparable property. Returns null if not set.
    // `i` goes up to `numComparableProperties()`, exclusive.
    [[nodiscard]] virtual std::optional<ComparisonTolerance> getComparisonTolerence( std::size_t i ) const = 0;

    // Sets the tolerance for a specific comparable property.
    // `i` goes up to `numComparableProperties()`, exclusive.
    virtual void setComparisonTolerance( std::size_t i, std::optional<ComparisonTolerance> newTolerance ) = 0;

    // If true, indicates that the getter will always return zero negative tolerance, and the setter will ignore the negative tolerance.
    // `i` goes up to `numComparableProperties()`, exclusive.
    [[nodiscard]] virtual bool comparisonToleranceIsAlwaysOnlyPositive( std::size_t i ) const { (void)i; return false; }


    // Reference values:

    // The number and types of reference values can be entirely different compared to `numComparableProperties()`.
    [[nodiscard]] virtual std::size_t numComparisonReferenceValues() const { return numComparableProperties(); }

    // `i` goes up to `numComparisonReferenceValues()`, exclusive.
    [[nodiscard]] virtual std::string_view getComparisonReferenceValueName( std::size_t i ) const = 0;

    // This can't be `std::optional<Var>`, because we still need the variant to know the correct type.
    struct ComparisonReferenceValue
    {
        using Var = std::variant<float, Vector3f>;

        bool isSet = false;

        // If `isSet == false`, this will hold zeroes, or some other default values.
        Var var;
    };

    // Returns the internal reference value.
    // If the value wasn't set yet (as indicated by `isSet == false`), you can still use the returned variant to get the expected type.
    // `i` goes up to `numComparisonReferenceValues()`, exclusive.
    [[nodiscard]] virtual ComparisonReferenceValue getComparisonReferenceValue( std::size_t i ) const = 0;
    // Sets the internal reference value. Makes `hasComparisonReferenceValue()` return true.
    // If you pass nullopt, removes this reference value.
    // Only a certain variant type is legal to pass, depending on the derived class and the index. Use `getComparisonReferenceValue()` to determine that type.
    // `i` goes up to `numComparisonReferenceValues()`, exclusive.
    virtual void setComparisonReferenceValue( std::size_t i, std::optional<ComparisonReferenceValue::Var> value ) = 0;
};

}
