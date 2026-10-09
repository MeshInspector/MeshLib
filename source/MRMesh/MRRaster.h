#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRImage.h"
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
    /// number of values along each axis: x is the width, y is the height, z is the number of layers
    Vector3i dims;

    /// type of all values
    ScalarType type = ScalarType::Unknown;

    /// returns the size of all values in bytes, or the largest size_t value if the size does not fit in size_t, which no data can have
    [[nodiscard]] MRMESH_API size_t dataSize() const;

    bool operator==( const RasterInfo& ) const = default;
};

/// regular grid of values of any type: an image if it has one layer, otherwise a volume
struct Raster
{
    RasterInfo info;

    /// values of type info.type: x changes first, then y (the rows go from top to bottom, unlike in Image), then z
    std::vector<uint8_t> data;
};

/// converts a raster with one layer to an image (the rows are reordered from bottom to top):
/// * RGBA8 values are taken as is, RGB8 values become opaque colors;
/// * UInt8 values become gray colors, and so do the high bytes of UInt16 values;
/// * the values of other types are scaled from the range of finite values to gray colors, NaN and infinite values become black
MRMESH_API Expected<Image> convertRasterToImage( const Raster& raster );

/// converts an image to a raster with one layer of RGBA8 values
MRMESH_API Expected<Raster> convertImageToRaster( const Image& image );

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
