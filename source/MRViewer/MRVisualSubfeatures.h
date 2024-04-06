#pragma once

#include "exports.h"

#include <MRMesh/MRSubfeatures.h>

namespace MR::Features
{

// This is similar to `Features::forEachSubfeature`, but slightly adjusted to be suitable for visualization.
MRVIEWER_API void forEachVisualSubfeature( const Features::Primitives::Variant& params, const Features::SubfeatureFunc& func );

} // namespace MR::Features
