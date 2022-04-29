#pragma once
#include "MRMeshFwd.h"
#include <string>
#include <typeinfo>

namespace MR
{

/** 
 * Finds the substring in the string.
 * \return position, npos if not found
 * \ingroup BasicGroup
 */
MRMESH_API size_t findSubstringCaseInsensitive( const std::string& string, const std::string& substring );

}
