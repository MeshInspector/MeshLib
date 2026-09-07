#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_XML
#include "exports.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRLinesLoadSettings.h"

#include <filesystem>

namespace MR::LinesLoad
{

/// loads shapes (polylines, paths, etc.) from file in .SVG format
MRIOEXTRAS_API Expected<Polyline3> fromSvg( const std::filesystem::path& file, const LinesLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<Polyline3> fromSvg( std::istream& in, const LinesLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<Polyline3> fromSvg( const char* data, size_t size, const LinesLoadSettings& settings = {} );

} // namespace MR::LinesLoad
#endif
