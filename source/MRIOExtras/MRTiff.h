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
/// * gray images with one sample per pixel keep the stored values, RGB images with interleaved samples become RGB8 or RGBA8 colors;
///   unlike the decoded colors, the stored values are not reordered according to the Orientation tag;
/// * libtiff decodes the other formats (palette, gray with alpha, YCbCr, CMYK, less than 8 bits per sample, etc.) to RGBA8 colors;
/// * the formats that libtiff cannot decode are read as stored too, as gray values or as colors depending on the number of samples
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

/// loads the first image of a .tiff file: libtiff decodes the formats it supports,
/// the others are loaded by RasterLoad::fromTiff and converted by convertRasterToImage
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
