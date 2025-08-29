#pragma once

#include "MRMesh/MRMeshFwd.h"
#include <optional>

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

    [[nodiscard]] virtual std::string_view getComparablePropertyName( std::size_t i ) const = 0;

    // Effectively computes `this->property - other.property`, or returns null on failure.
    // `other` will be `dynamic_cast`-ed to the same derived type as `this`.
    [[nodiscard]] virtual std::optional<float> compareProperty( const Object& other, std::size_t i ) const = 0;


    // Tolerances:

    struct ComparisonTolerance
    {
        // How much larger can this value be compared to the reference?
        float positive = 0;
        // How much smaller can this value be compared to the reference?
        // This number should normally be zero or negative.
        float negative = 0;
    };

    // True if this object includes tolerance information.
    [[nodiscard]] virtual bool hasComparisonTolerances() const = 0;

    // Returns the tolerance for a specific comparable property. Only call if `hasComparisonTolerances() == true`.
    [[nodiscard]] virtual ComparisonTolerance getComparisonTolerences( std::size_t i ) const = 0;

    // Sets the tolerance for a specific comparable property. Only call if `hasComparisonTolerances() == true`.
    // Makes `hasComparisonTolerances()` return true.
    virtual void setComparisonTolerance( std::size_t i, const ComparisonTolerance& newTolerance ) = 0;

    // Removes all tolerance information, and makes `hasComparisonTolerances()` return true.
    virtual void resetComparisonTolerances() = 0;


    // Reference values:

    // True if this object has built-in reference values, and doesn't need another object to compare against.
    [[nodiscard]] virtual bool hasComparisonReferenceValues() const = 0;
    // Returns the internal reference value. Only call if `hasComparisonReferenceValue() == true`.
    [[nodiscard]] virtual float getComparisonReferenceValue( std::size_t i ) const = 0;
    // Sets the internal reference value. Makes `hasComparisonReferenceValue()` return true.
    virtual void setComparisonReferenceValue( std::size_t i, float value ) = 0;
    // Makes `hasComparisonReferenceValue()` return false.
    virtual void resetComparisonReferenceValues() = 0;
};

}
