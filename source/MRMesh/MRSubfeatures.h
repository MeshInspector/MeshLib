#pragma once

#include "MRMesh/MRFeatures.h"

#include <functional>

namespace MR::Features
{

// Describes a single feature produced by `forEachSubfeature()`.
struct SubfeatureInfo
{
    // A user-friendly name.
    std::string_view name;

    // Whether the feature has infinite length.
    bool isInfinite = false;

    // Call this to create this subfeature.
    // Pass the same feature you passed to `forEachSubfeature()`, or a DIFFERENT one to create the same kind of subfeature relative to that.
    std::function<Primitives::Variant( const Features::Primitives::Variant& feature )> create;
};

// A user callback for `forEachSubfeature()`.
using SubfeatureFunc = std::function<void( const SubfeatureInfo& info )>;

// Decomposes a feature to its subfeatures, by calling `func()` on each subfeature.
// This only returns the direct subfeatures. You can call this recursively to obtain all features,
//   but beware of duplicates (there's no easy way to filter them).
MRMESH_API void forEachSubfeature( const Features::Primitives::Variant& feature, const SubfeatureFunc& func );

}
