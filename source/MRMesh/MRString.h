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


/**
 * Splits given string by delimiter.
 * \return vector of split strings
 * \ingroup BasicGroup
 */
MRMESH_API std::vector<std::string> split( const std::string& string, char delimiter );

}
