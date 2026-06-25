#pragma once
#include "MRPch/MRBindingMacros.h"
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRDistanceMapParams.h"
#include "MRExpected.h"
#include <filesystem>

namespace MR
{

namespace DistanceMapSave
{

/// \defgroup DistanceMapSaveGroup DistanceMap Save
/// \ingroup IOGroup
/// \{

/**
 * @brief Save DistanceMap to binary file
 * Format:
 * 2 integer - DistanceMap.resX & DistanceMap.resY
 * [resX * resY] float - matrix of values
 */
MRMESH_API Expected<void> toRAW( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings = {} );
[[deprecated( "Use toRAW( dmap, path, settings )")]]
MR_BIND_IGNORE inline Expected<void> toRAW( const std::filesystem::path& path, const DistanceMap& dmap )
{
    return toRAW( dmap, path );
}

MRMESH_API Expected<void> toMrDistanceMap( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings = {} );
[[deprecated( "Use toMrDistanceMap( dmap, path, settings )")]]
MR_BIND_IGNORE inline Expected<void> toMrDistanceMap( const std::filesystem::path& path, const DistanceMap& dmapObject, const DistanceMapToWorld& params )
{
    const auto xf = params.xf();
    return toMrDistanceMap( dmapObject, path, {
        .xf = &xf,
    } );
}

MRMESH_API Expected<void> toAnySupportedFormat( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings = {} );
[[deprecated( "Use toAnySupportedFormat( dmap, path, settings )")]]
MR_BIND_IGNORE inline Expected<void> toAnySupportedFormat( const std::filesystem::path& path, const DistanceMap& dmapObject, const AffineXf3f * xf = nullptr )
{
    return toAnySupportedFormat( dmapObject, path, {
        .xf = xf,
    } );
}

/// \}

} // namespace DistanceMapSave

} // namespace MR
