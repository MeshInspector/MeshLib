#pragma once
#include "MRStringConvert.h"
#include <functional>
#include <string>
#include <filesystem>

namespace MR
{

/// Argument value - progress in [0,1];
/// returns true to continue the operation and returns false to stop the operation
/// \ingroup BasicStructuresGroup
typedef std::function<bool( float )> ProgressCallback;

// Returns message showed when loading is canceled
inline std::string getCancelMessage( const std::filesystem::path& path )
{
	return "Loading canceled: " + utf8string( path );
}

}
