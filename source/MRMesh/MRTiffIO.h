#pragma once
#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_TIFF )
#include "MRExpected.h"
#include "MRVector2.h"
#include <filesystem>
#include <string>

namespace MR
{

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
MRMESH_API bool isTIFFFile( const std::filesystem::path& path );

// reads parameters of tiff file
MRMESH_API Expected<TiffParameters> readTiffParameters( const std::filesystem::path& path );

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

// load values from tiff to ouput.data
MRMESH_API Expected<void> readRawTiff( const std::filesystem::path& path, RawTiffOutput& output );

// writes bytes to tiff file
MRMESH_API Expected<void> writeRawTiff( const uint8_t* bytes, const std::filesystem::path& path,
    const BaseTiffParameters& params );

}

#endif