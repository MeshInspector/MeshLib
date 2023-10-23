#pragma once

#include "MRMeshFwd.h"
#include <cstdint>
#include <string>
#include <vector>

namespace MR
{


/// \defgroup IOGroup Input/Output
/// \{

/// \defgroup Base64Group Base64
/// \{

/// encodes binary data into textual Base64 format
MRMESH_API std::string encode64( const std::uint8_t * data, size_t size );

/// decodes Base64 format into binary data
MRMESH_API std::vector<std::uint8_t> decode64( const std::string &val );

/// \}
/// \}

} // namespace MR
