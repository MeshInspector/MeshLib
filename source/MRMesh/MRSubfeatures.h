#pragma once

#include "MRMesh/MRFeatures.h"
#include "MRMesh/MRFlagOperators.h"

#include <functional>

namespace MR
{

enum class OfferedSubfeatureFlags
{
    // This subfeature can also be reached indirectly through a different subfeature.
    // It doesn't need to be displayed in the GUI to save space, but you can still display a 3D object for it for convenience.
    indirect = 1 << 0,

    // Visualizing this subfeature in 3D would be intrusive. Don't create a preview object by default.
    visualizationIsIntrusive = 1 << 1,
};
MR_MAKE_FLAG_OPERATORS( OfferedSubfeatureFlags )

using CreateSubfeatureFunc = std::function<Features::Primitives::Variant()>;
using OfferSubfeatureFunc = std::function<void( std::string_view name, OfferedSubfeatureFlags flags, const CreateSubfeatureFunc& create )>;

// Decomposes a feature to its subfeatures.
MRMESH_API void extractSubfeatures( const Features::Primitives::Variant& feature, const OfferSubfeatureFunc& offerSubFeature );

}
