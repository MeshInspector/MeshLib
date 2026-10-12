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
/// * gray, palette and RGB images keep the stored samples of any type (gray values, color indices or color components) as channels;
///   unlike the decoded colors, the stored samples are not reordered according to the Orientation tag;
/// * libtiff decodes the other formats (gray with alpha, YCbCr, CMYK, less than 8 bits per sample, etc.) to four UInt8 channels
///   of RGBA colors;
/// * the formats that libtiff cannot decode are read as stored too if each pixel has one sample or three or more interleaved samples
///   of 8 to 64 bits, otherwise loading fails;
/// * the files compressed by the methods that libtiff does not support fail to load
MRIOEXTRAS_API Expected<Raster> fromTiff( const std::filesystem::path& path, const RasterLoadSettings& settings = {} );

/// loads everything about the raster except its values from .tiff format
MRIOEXTRAS_API Expected<RasterInfo> infoFromTiff( const std::filesystem::path& path );

} // namespace RasterLoad

namespace RasterSave
{

/// saves a raster with one layer in .tiff format: one channel of gray values, or three or four channels of RGB or RGBA colors,
/// of any type from UInt8 to Float64
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

} // namespace DistanceMapSave

} // namespace MR
#endif
