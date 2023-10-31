#pragma once
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

MRMESH_API extern const IOFilters Filters;

/**
 * @brief Save DistanceMap to binary file
 * Format: 
 * 2 integer - DistanceMap.resX & DistanceMap.resY
 * [resX * resY] float - matrix of values
 */
MRMESH_API VoidOrErrStr toRAW( const std::filesystem::path& path, const DistanceMap& dmap );
MRMESH_API VoidOrErrStr toMrDistanceMap( const std::filesystem::path& path, const DistanceMap& dmapObject, const DistanceMapToWorld& params );
MRMESH_API VoidOrErrStr toAnySupportedFormat( const std::filesystem::path& path, const DistanceMap& dmapObject, const DistanceMapToWorld* params = nullptr );

/// \}

} // namespace DistanceMapSave

} // namespace MR
