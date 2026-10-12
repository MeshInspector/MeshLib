#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRScalarConvert.h"
#include "MRVector3.h"

#include <vector>

namespace MR
{

/// \defgroup RasterGroup Raster
/// \ingroup BasicStructuresGroup
/// \{

/// everything about a raster except its values
struct RasterInfo
{
    /// number of pixels along each axis: x is the width, y is the height, z is the number of layers
    Vector3i dims;

    /// type of all values
    ScalarType type = ScalarType::Unknown;

    /// number of values in each pixel, e.g. 1 for gray values, 3 for RGB colors, 4 for RGBA colors, or any number of bands;
    /// convertRasterToImage takes three or more channels as RGB(A) colors, so three or four gray bands look like colors there
    int channels = 1;

    /// returns the size of all values in bytes, or the largest size_t value if the size does not fit in size_t, which no data can have
    [[nodiscard]] MRMESH_API size_t dataSize() const;

    bool operator==( const RasterInfo& ) const = default;
};

/// regular grid of values of any type: an image if it has one layer, otherwise a volume
struct Raster
{
    RasterInfo info;

    /// values of type info.type: the channels of a pixel go together, x changes first, then y (the rows go from top to bottom,
    /// unlike in Image), then z
    std::vector<uint8_t> data;
};

/// converts a raster with one layer to an image (the rows are reordered from bottom to top):
/// * the values of one channel are scaled from the range of finite values to gray colors, NaN and infinite values become black;
/// * three or more channels are RGB or RGBA colors (the channels after the fourth are ignored): integer values keep their high byte
///   and are taken as unsigned ones, floating-point values are clamped to [0, 1], NaN becomes 0
MRMESH_API Expected<Image> convertRasterToImage( const Raster& raster );

/// converts the values of a raster to floats: dst is resized to dims.x * dims.y * dims.z values (memory is reallocated only if its
/// capacity is too small), in the order of the raster:
/// * one channel is taken as is;
/// * three or more channels give the luma 0.299 R + 0.587 G + 0.114 B of the first three;
/// fails for two channels, an unknown type, or data whose size does not match the dimensions
MRMESH_API Expected<void> convertRasterToFloats( const Raster& raster, std::vector<float>& dst );

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
