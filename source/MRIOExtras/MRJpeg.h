#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_JPEG
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRImage.h>

#include <filesystem>

namespace MR
{

namespace ImageLoad
{

/// loads from .jpg format
MRIOEXTRAS_API Expected<Image> fromJpeg( const std::filesystem::path& path );
MRIOEXTRAS_API Expected<Image> fromJpeg( std::istream& in );
MRIOEXTRAS_API Expected<Image> fromJpeg( const char* data, size_t size );

} // namespace ImageLoad

namespace ImageSave
{

/// saves in .jpg format
MRIOEXTRAS_API Expected<void> toJpeg( const Image& image, const std::filesystem::path& path );

} // namespace ImageSave

} // namespace MR
#endif
