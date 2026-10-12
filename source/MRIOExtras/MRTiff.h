#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_TIFF
#include "exports.h"

#include <MRMesh/MRDistanceMap.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRRaster.h>
#include <MRMesh/MRVector2.h>

#include <filesystem>
#include <string>

namespace MR
{

namespace RasterLoad
{

/// loads the first image of a .tiff file as a raster with one layer:
/// * gray, palette and RGB images keep the stored samples of any type (gray values, color indices or color components) as channels;
/// * libtiff decodes the other formats (gray with alpha, YCbCr, CMYK, less than 8 bits per sample, etc.) to four UInt8 channels
///   of RGBA colors;
/// * the formats that libtiff cannot decode are read as stored too if each pixel has one sample or three or more interleaved samples
///   of 8 to 64 bits, otherwise loading fails;
/// * the files compressed by the methods that libtiff does not support fail to load;
/// * the rows keep the stored order whatever the Orientation tag says, also in the decoded formats
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

namespace DistanceMapLoad
{

/// loads the first image of a .tiff file: one channel is taken as is, three or more channels give the luma of the first three,
/// see convertRasterToFloats; settings.distanceMapToWorld gets the GeoTIFF transformation, or the identity if the file has none
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

// raw TIFF API: reads and writes the samples in the caller's buffers with the GeoTIFF transformation; unlike the Raster API,
// it reports the tile layout and writes a GDAL no-data value

struct BaseTiffParameters
{
    enum class SampleType
    {
        Unknown,
        Uint,
        Int,
        Float
    } sampleType{ SampleType::Unknown };

    enum class ValueType
    {
        Unknown,
        Scalar,
        RGB,
        RGBA,
    } valueType{ ValueType::Unknown };

    // size of internal data in file
    int bytesPerSample = 0;

    // size of image if not layered, otherwise size of layer
    Vector2i imageSize;

    bool operator==( const BaseTiffParameters& ) const = default;
};

struct TiffParameters : BaseTiffParameters
{
    // true if tif file is tiled
    bool tiled = false;
    Vector2i tileSize;
    int layers = 1;
    // tile depth (if several layers)
    int depth = 0;

    bool operator==( const TiffParameters& ) const = default;
};

// returns true if given file is tiff
MRIOEXTRAS_API bool isTIFFFile( const std::filesystem::path& path );

// reads parameters of tiff file: the formats that RasterLoad::fromTiff decodes are 8-bit RGBA ones
MRIOEXTRAS_API Expected<TiffParameters> readTiffParameters( const std::filesystem::path& path );

struct RawTiffOutput
{
    // main output data, should be allocated
    uint8_t* bytes{ nullptr };
    // allocated data size
    size_t size{ 0 };
    // optional params output
    TiffParameters* params{ nullptr };
    // optional pixel to world transform
    AffineXf3f* p2wXf{ nullptr };
    // input if true loads tiff file as floats array
    bool convertToFloat{ true };
    // min max
    float* min{ nullptr };
    float* max{ nullptr };
};

// load values from tiff to output.bytes: the samples as RasterLoad::fromTiff reads them, or floats as DistanceMapLoad::fromTiff reads them;
// fails if the output is too small
MRIOEXTRAS_API Expected<void> readRawTiff( const std::filesystem::path& path, RawTiffOutput& output );

struct WriteRawTiffParams
{
    BaseTiffParameters baseParams;
    // optional transformation data written to GeoTIFF's ModelTransformationTag
    const AffineXf3f* xf = nullptr;
    // optional NoData value written to GDAL_NODATA
    std::string noData;
};

// writes bytes to tiff file
MRIOEXTRAS_API Expected<void> writeRawTiff( const uint8_t* bytes, const std::filesystem::path& path,
    const WriteRawTiffParams& params );
[[deprecated( "use WriteRawTiffParams version instead" )]]
inline Expected<void> writeRawTiff( const uint8_t* bytes, const std::filesystem::path& path,
    const BaseTiffParameters& params, const AffineXf3f* xf )
{
    return writeRawTiff( bytes, path, { .baseParams = params, .xf = xf } );
}

} // namespace MR
#endif
