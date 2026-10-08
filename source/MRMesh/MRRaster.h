#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRColor.h"
#include "MRExpected.h"
#include "MRImage.h"
#include "MRScalarConvert.h"
#include "MRVector2.h"

#include <optional>
#include <vector>

namespace MR
{

/// \defgroup RasterGroup Raster
/// \ingroup BasicStructuresGroup
/// \{

/// everything about a raster image except its pixel values
struct RasterInfo
{
    /// width and height in pixels
    Vector2i resolution;

    /// number of samples per pixel: 1 - gray value or palette index, 2 - gray value and alpha, 3 - RGB, 4 - RGBA;
    /// the channels after the fourth one have no predefined meaning
    int channels = 1;

    /// type of all samples
    ScalarType sampleType = ScalarType::Unknown;

    /// colors of the indices of a single-channel raster with integer samples; empty if the raster has no palette
    std::vector<Color> palette;

    /// true if the smallest gray value means white and the largest one black (TIFF's PHOTOMETRIC_MINISWHITE)
    bool minIsWhite = false;

    /// optional transformation of (column, row, sample value) to world coordinates, e.g. from GeoTIFF tags
    std::optional<AffineXf3f> pixelToWorld;

    /// optional value of the samples without valid data, e.g. from GDAL_NODATA tag
    std::optional<double> noData;

    /// returns the size of one sample in bytes, or 0 if sampleType cannot be used in rasters
    [[nodiscard]] MRMESH_API size_t sampleSize() const;

    /// returns the size of all samples in bytes
    [[nodiscard]] MRMESH_API size_t dataSize() const;
};

/// raster image with samples of any type and any number of channels
struct Raster
{
    RasterInfo info;

    /// samples of type info.sampleType: the rows go from top to bottom (unlike in Image), the pixels of a row from left to right,
    /// and the samples of a pixel are stored together
    std::vector<uint8_t> data;
};

/// converts a raster to an image (the rows are reordered from bottom to top):
/// * 8-bit samples are taken as is, 16-bit samples are scaled to 8 bits;
/// * the samples of other types are scaled from the range of valid values (neither noData nor NaN) to [0, 255] if they are gray,
///   the invalid ones become black; color and alpha samples are clamped to [0, 1] for floating-point types and to [0, 255] for integers;
/// * gray values are inverted if minIsWhite is set;
/// * palette indices are replaced with palette colors, the indices outside of the palette become black
MRMESH_API Expected<Image> convertRasterToImage( const Raster& raster );

/// converts an image to a raster with four 8-bit channels
MRMESH_API Raster convertImageToRaster( const Image& image );

/// settings for loading rasters from external formats
struct RasterLoadSettings
{
    /// to report load progress and cancel loading if user desires
    ProgressCallback progress;
};

/// settings for saving rasters in external formats
struct RasterSaveSettings
{
    /// to report save progress and cancel saving if user desires
    ProgressCallback progress;
};

/// \}

} // namespace MR
