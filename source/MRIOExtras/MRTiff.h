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

/// loads the first image of a .tiff file as a raster with one layer:
/// * gray images with one sample per pixel keep the stored values, even if the smallest value means white;
/// * RGB images become RGB8 colors, or RGBA8 colors with the fourth sample as alpha if they have more samples;
/// * other formats (palette, gray with more samples, YCbCr, CMYK, less than 8 bits per sample, etc.) are decoded by libtiff to RGBA8 colors;
/// * the formats that libtiff cannot decode keep the stored values of the first sample of each pixel
MRIOEXTRAS_API Expected<Raster> fromTiff( const std::filesystem::path& path, const RasterLoadSettings& settings = {} );

/// loads everything about the raster except its values from .tiff format
MRIOEXTRAS_API Expected<RasterInfo> infoFromTiff( const std::filesystem::path& path );

} // namespace RasterLoad

namespace RasterSave
{

/// saves a raster with one layer in .tiff format; all value types except Float32_4 are supported
MRIOEXTRAS_API Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings = {} );

} // namespace RasterSave

namespace ImageLoad
{

/// loads the first image of a .tiff file: converts the raster loaded by RasterLoad::fromTiff to an image as convertRasterToImage does,
/// but 8-bit and 16-bit gray is inverted if the smallest value means white
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
