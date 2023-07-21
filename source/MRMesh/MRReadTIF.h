#pragma once
#ifndef MRMESH_NO_TIFF
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRVector2.h"
#include <filesystem>
#include <string>

namespace MR
{

struct TifParameters
{
    enum class SampleType
    {
        Uint,
        Int,
        Float,
        Unknown
    } sampleType{ SampleType::Unknown };

    enum class ValueType 
    {
        Scalar,
        RGB,
        RGBA,
        Unknown
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

    bool operator==( const TifParameters& ) const = default;
};

// returns true if given file is tif
bool isTIFFile( const std::filesystem::path& path );

// reads parameters of tif file
Expected<TifParameters, std::string> readTifParameters( const std::filesystem::path& path );

struct RawTifOutput
{
    // main output data, should be allocated
    float* data{ nullptr };
    // parameters to compare if reading series of files
    const TifParameters* params{ nullptr };
    // min max
    float* min{ nullptr };
    float* max{ nullptr };
};

// 
VoidOrErrStr readRawTif( const std::filesystem::path& path, RawTifOutput& output );

}

#endif