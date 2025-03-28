#pragma once

#include "MRMeshFwd.h"
#include "MRDistanceMap.h"
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"

#include <filesystem>

namespace MR
{

namespace DistanceMapLoad
{
/// \defgroup DistanceMapLoadGroup DistanceMap Load
/// \ingroup IOGroup
/// \{

/**
 * @brief Load DistanceMap from binary file
 * Format:
 * 2 integer - DistanceMap.resX & DistanceMap.resY
 * [resX * resY] float - matrix of values
 */
MRMESH_API Expected<DistanceMap> fromRaw( const std::filesystem::path& path, const DistanceMapLoadSettings& settings = {} );
[[deprecated( "Use fromRaw( path, settings )")]]
inline Expected<DistanceMap> fromRaw( const std::filesystem::path& path, ProgressCallback progressCb )
{
    return fromRaw( path, DistanceMapLoadSettings {
        .progress = progressCb,
    } );
}

MRMESH_API Expected<DistanceMap> fromMrDistanceMap( const std::filesystem::path& path, const DistanceMapLoadSettings& settings = {} );
[[deprecated( "Use fromMrDistanceMap( path, settings )")]]
inline Expected<DistanceMap> fromMrDistanceMap( const std::filesystem::path& path, DistanceMapToWorld& params, ProgressCallback progressCb = {} )
{
    return fromMrDistanceMap( path, {
        .distanceMapToWorld = &params,
        .progress = progressCb,
    } );
}

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
MRMESH_API Expected<DistanceMap> fromTiff( const std::filesystem::path& path, const DistanceMapLoadSettings& settings = {} );
[[deprecated( "Use fromTiff( path, settings )")]]
inline Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld& params, ProgressCallback progressCb = {} )
{
    return fromTiff( path, {
        .distanceMapToWorld = &params,
        .progress = progressCb,
    } );
}
#endif

MRMESH_API Expected<DistanceMap> fromAnySupportedFormat( const std::filesystem::path& path, const DistanceMapLoadSettings& settings = {} );
[[deprecated( "Use fromAnySupportedFormat( path, settings )" )]]
inline Expected<DistanceMap> fromAnySupportedFormat( const std::filesystem::path& path, DistanceMapToWorld* params, ProgressCallback progressCb = {} )
{
    return fromAnySupportedFormat( path, {
        .distanceMapToWorld = params,
        .progress = progressCb,
    } );
}

/// \}

} // namespace DistanceMapLoad

} // namespace MR
