#pragma once
#include "MRMeshFwd.h"
#include <string>
#include <typeinfo>

namespace MR
{

// Finds the substring in the string. 
// Returns position, npos if not found
MRMESH_API size_t findSubstringCaseInsensitive( const std::string& string, const std::string& substring );

}
