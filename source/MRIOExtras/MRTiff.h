#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_TIFF
#include "exports.h"

#include <MRMesh/MRDistanceMap.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRRaster.h>

#include <filesystem>
#include <optional>

namespace MR
{

namespace RasterLoad
{

/// TIFF-specific raster load parameters
struct TiffLoadSettings
{
    /// optional output: the value meaning no data (GDAL_NODATA) if the file has one, otherwise nullopt
    std::optional<double>* noData = nullptr;
};

/// loads the first image of a .tiff file as a raster with one layer:
/// * gray, palette and RGB images keep the stored samples of any type (gray values, color indices or color components) as channels;
/// * libtiff decodes the other formats (gray with alpha, YCbCr, CMYK, less than 8 bits per sample, etc.) to four UInt8 channels
///   of RGBA colors, which are not multiplied by alpha unless the file stores them so (associated alpha);
/// * the formats that libtiff cannot decode are read as stored too if each pixel has one sample or three or more interleaved samples
///   of 8 to 64 bits, otherwise loading fails;
/// * the files compressed by the methods that libtiff does not support fail to load;
/// * the rows keep the stored order whatever the Orientation tag says, also in the decoded formats
MRIOEXTRAS_API Expected<Raster> fromTiff( const std::filesystem::path& path, const RasterLoadSettings& settings = {},
    const TiffLoadSettings& tiffSettings = {} );

/// loads everything about the raster except its values from .tiff format
MRIOEXTRAS_API Expected<RasterInfo> infoFromTiff( const std::filesystem::path& path );

} // namespace RasterLoad

/// settings for saving rasters in .tiff format
struct TiffSaveSettings : RasterSaveSettings
{
    /// optional value meaning no data, written as GDAL_NODATA
    std::optional<double> noData;
};

namespace RasterSave
{

/// saves a raster with one layer in .tiff format: one channel of gray values, or three or four channels of RGB or RGBA colors
/// (unassociated alpha: the colors are not multiplied by it), of any type from UInt8 to Float64
MRIOEXTRAS_API Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const TiffSaveSettings& settings );
MRIOEXTRAS_API Expected<void> toTiff( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings = {} );

} // namespace RasterSave

namespace ImageLoad
{

/// loads the first image of a .tiff file: libtiff decodes the formats it supports,
/// the others are loaded by RasterLoad::fromTiff and converted by convertRasterToImage;
/// the colors are not multiplied by alpha unless the file stores them so (associated alpha)
MRIOEXTRAS_API Expected<Image> fromTiff( const std::filesystem::path& path );

} // namespace ImageLoad

namespace ImageSave
{

/// saves in .tiff format with unassociated alpha, as in Image: the colors are not multiplied by it
MRIOEXTRAS_API Expected<void> toTiff( const Image& image, const std::filesystem::path& path );

} // namespace ImageSave

namespace DistanceMapLoad
{

/// loads the first image of a .tiff file: one channel is taken as is, three or more channels give the luma of the first three,
/// see convertRasterToFloats; the values of one channel equal to the value meaning no data (GDAL_NODATA) are invalid;
/// settings.distanceMapToWorld gets the GeoTIFF transformation, or the identity if the file has none
MRIOEXTRAS_API Expected<DistanceMap> fromTiff( const std::filesystem::path& path, const DistanceMapLoadSettings& settings = {} );
[[deprecated( "Use fromTiff( path, settings )")]]
MR_BIND_IGNORE inline Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld& params, ProgressCallback progressCb = {} )
{
    return fromTiff( path, {
        .distanceMapToWorld = &params,
        .progress = progressCb,
    } );
}

} // namespace DistanceMapLoad

namespace DistanceMapSave
{

/// saves in .tiff format
MRIOEXTRAS_API Expected<void> toTiff( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings = {} );

} // namespace DistanceMapSave

} // namespace MR
#endif
