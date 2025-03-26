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
MRMESH_API Expected<DistanceMap> fromRaw( const std::filesystem::path& path, DistanceMapToWorld* params, ProgressCallback progressCb = {} );
[[deprecated( "Use fromRaw( path, params, progressCb )")]]
inline Expected<DistanceMap> fromRaw( const std::filesystem::path& path, ProgressCallback progressCb = {} )
{
    return fromRaw( path, nullptr, progressCb );
}

MRMESH_API Expected<DistanceMap> fromMrDistanceMap( const std::filesystem::path& path, DistanceMapToWorld* params = nullptr, ProgressCallback progressCb = {} );
[[deprecated( "Use fromMrDistanceMap( path, params, progressCb )")]]
inline Expected<DistanceMap> fromMrDistanceMap( const std::filesystem::path& path, DistanceMapToWorld& params, ProgressCallback progressCb = {} )
{
    return fromMrDistanceMap( path, &params, progressCb );
}

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
MRMESH_API Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld* params, ProgressCallback progressCb = {} );
[[deprecated( "Use fromMrDistanceMap( path, params, progressCb )")]]
inline Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld& params, ProgressCallback progressCb = {} )
{
    return fromTiff( path, &params, progressCb );
}
#endif

MRMESH_API Expected<DistanceMap> fromAnySupportedFormat( const std::filesystem::path& path, DistanceMapToWorld* params, ProgressCallback progressCb = {} );

/// \}

} // namespace DistanceMapLoad

} // namespace MR
