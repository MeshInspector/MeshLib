#pragma once

#include "MRMesh/MRObjectComparableWithReference.h"
#include "exports.h"

// Specialized UI for quality-control things.

namespace MR::QualityControl
{

// An input widget for the tolerance. Uses `inputPlusMinus()` internally.
// NOTE: This doesn't include the "reset" button to remove the tolerance, must draw that manually.
MRVIEWER_API bool inputTolerance( const char* label, std::optional<ObjectComparableWithReference::ComparisonTolerance>& toleranceOpt );

}
