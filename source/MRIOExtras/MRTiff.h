#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_TIFF
#include "exports.h"

#include <MRMesh/MRDistanceMap.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRImage.h>

#include <filesystem>

namespace MR
{

namespace DistanceMapLoad
{

/// loads from .tiff format
MRIOEXTRAS_API Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld* dmapToWorld = nullptr );

} // namespace DistanceMapLoad

namespace DistanceMapSave
{

/// saves to .tiff format
MRIOEXTRAS_API Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, DistanceMapToWorld* dmapToWorld = nullptr );

} // namespace DistanceMapSave

namespace ImageLoad
{

/// loads from .tiff format
MRIOEXTRAS_API Expected<Image> fromTiff( const std::filesystem::path& path );

} // namespace ImageLoad

namespace ImageSave
{

/// saves in .tiff format
MRIOEXTRAS_API Expected<void> toTiff( const Image& image, const std::filesystem::path& path );

} // namespace ImageSave

} // namespace MR
#endif
