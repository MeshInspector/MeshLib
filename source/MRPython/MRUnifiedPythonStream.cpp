#include "MRUnifiedPythonStream.h"
#include <mutex>

namespace MR
{
static std::mutex sMutex;
static std::string sText;

void UnifiedPythonStream::append( std::string_view text )
{
    std::lock_guard lock( sMutex );
    sText += text;
}

std::string UnifiedPythonStream::str()
{
    std::lock_guard lock( sMutex );
    return sText;
}

void UnifiedPythonStream::clear()
{
    std::lock_guard lock( sMutex );
    sText.clear();
}
}
