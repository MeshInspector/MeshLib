#pragma once

#include "MRMesh/MRObjectComparableWithReference.h"
#include "exports.h"

// Specialized UI for quality-control things.

namespace MR::QualityControl
{

// An input widget for the tolerance. Uses `inputPlusMinus()` internally.
// NOTE: This doesn't include the "reset" button to remove the tolerance, must draw that manually.
// `index` is the same index as used by `ObjectComparableWithReference::getComparisonTolerence()`, see that for more details.
MRVIEWER_API bool inputTolerance( const char* label, ObjectComparableWithReference& object, std::size_t index );

}
