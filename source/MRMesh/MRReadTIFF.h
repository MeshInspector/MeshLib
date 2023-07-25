#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_TIFF )
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRVector2.h"
#include <filesystem>
#include <string>

namespace MR
{

struct TiffParameters
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

    // true if tif file is tiled
    bool tiled = false;
    Vector2i tileSize;
    int layers = 1;
    // tile depth (if several layers)
    int depth = 0;

    // size of image if not layered, otherwise size of layer
    Vector2i imageSize;

    bool operator==( const TiffParameters& ) const = default;
};

// returns true if given file is tiff
bool isTIFFFile( const std::filesystem::path& path );

// reads parameters of tiff file
Expected<TiffParameters, std::string> readTiffParameters( const std::filesystem::path& path );

struct RawTiffOutput
{
    // main output data, should be allocated
    float* data{ nullptr };
    // allocated data size
    size_t size{ 0 };
    // optional params output
    TiffParameters* params{ nullptr };
    // optional pixel to world transform
    AffineXf3f* p2wXf{ nullptr };
    // min max
    float* min{ nullptr };
    float* max{ nullptr };
};

// load values from tiff to ouput.data
VoidOrErrStr readRawTiff( const std::filesystem::path& path, RawTiffOutput& output );

}

#endif