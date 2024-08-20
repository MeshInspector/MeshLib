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
MRMESH_API VoidOrErrStr toAnySupportedFormat( const std::filesystem::path& path, const DistanceMap& dmapObject, const AffineXf3f * xf = nullptr );

/// \}

} // namespace DistanceMapSave

/// saves distance map to monochrome image in scales of gray:
/// \param threshold - threshold of maximum values [0.; 1.]. invalid pixel set as 0. (black)
/// minimum (close): 1.0 (white)
/// maximum (far): threshold
/// invalid (infinity): 0.0 (black)
MRMESH_API VoidOrErrStr saveDistanceMapToImage( const DistanceMap& distMap, const std::filesystem::path& filename, float threshold = 1.f / 255 );

} // namespace MR
