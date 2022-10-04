#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRDistanceMapParams.h"
#include <tl/expected.hpp>
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
 * @detail Format: 
 * 2 integer - DistanceMap.resX & DistanceMap.resY
 * [resX * resY] float - matrix of values
 */
MRMESH_API tl::expected<void, std::string> toRAW( const std::filesystem::path& path, const DistanceMap& dmap );
MRMESH_API tl::expected<void, std::string> toMrDistanceMap( const std::filesystem::path& path, const DistanceMap& dmapObject, const DistanceMapToWorld* params = nullptr );
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const std::filesystem::path& path, const DistanceMap& dmapObject, const DistanceMapToWorld* params = nullptr );

/// \}

} // namespace DistanceMapSave

} // namespace MR
