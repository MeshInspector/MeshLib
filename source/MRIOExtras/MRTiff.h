#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_TIFF
#include "exports.h"

#include <MRMesh/MRDistanceMapParams.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRRaster.h>

#include <filesystem>

namespace MR
{

namespace RasterLoad
{

/// loads from .tiff format; the samples are kept as stored in the file, except for the formats without plain sample values
/// (YCbCr, CMYK, CIE L*a*b*, less than 8 bits per sample, etc.), which are decoded to 8-bit RGBA;
/// PHOTOMETRIC_MINISWHITE sets RasterInfo::minIsWhite only for 8-bit and 16-bit samples
MRIOEXTRAS_API Expected<Raster> fromTiff( const std::filesystem::path& path, const RasterLoadSettings& settings = {} );

/// loads everything about the raster except its samples from .tiff format
MRIOEXTRAS_API Expected<RasterInfo> infoFromTiff( const std::filesystem::path& path );

} // namespace RasterLoad

namespace RasterSave
{

/// saves in .tiff format
MRIOEXTRAS_API Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings = {} );

} // namespace RasterSave

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

namespace DistanceMapSave
{

/// saves in .tiff format
MRIOEXTRAS_API Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings = {} );

} // namespace ImageSave

} // namespace MR
#endif
