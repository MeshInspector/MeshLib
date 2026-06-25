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
/// \param ignoreDecompressErrors if true do not return decompression error if the header was read sucessfully
MRIOEXTRAS_API Expected<Image> fromJpeg( const std::filesystem::path& path, bool ignoreDecompressErrors = false );
MRIOEXTRAS_API Expected<Image> fromJpeg( std::istream& in, bool ignoreDecompressErrors = false );
MRIOEXTRAS_API Expected<Image> fromJpeg( const char* data, size_t size, bool ignoreDecompressErrors = false );

} // namespace ImageLoad

namespace ImageSave
{

/// saves in .jpg format
MRIOEXTRAS_API Expected<void> toJpeg( const Image& image, const std::filesystem::path& path );

} // namespace ImageSave

} // namespace MR
#endif
