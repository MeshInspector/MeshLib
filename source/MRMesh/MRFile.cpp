#include "MRFile.h"
#include "MRStringConvert.h"

namespace MR
{

FILE * fopen( const std::filesystem::path & filename, const char * mode )
{
#ifdef _WIN32
    auto wpath = utf8ToWide( utf8string( filename ).c_str() );
    std::wstring wmode( mode, mode + std::strlen( mode ) );
    return _wfopen( wpath.c_str(), wmode.c_str() );
#else
    // Linux
    return std::fopen( filename.string().c_str(), mode );
#endif
}

} //namespace MR
