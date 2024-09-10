#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_PNG
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRImage.h>

#include <filesystem>

namespace MR
{

namespace ImageLoad
{

/// loads from .png format
MRIOEXTRAS_API Expected<Image> fromPng( const std::filesystem::path& path );
MRIOEXTRAS_API Expected<Image> fromPng( std::istream& in );

} // namespace ImageLoad

namespace ImageSave
{

/// saves in .png format
MRIOEXTRAS_API Expected<void> toPng( const Image& image, const std::filesystem::path& path );
MRIOEXTRAS_API Expected<void> toPng( const Image& image, std::ostream& os );

} // namespace ImageSave

} // namespace MR
#endif
