#include "MRFile.h"
#include "MRStringConvert.h"
#include <cstring>

namespace MR
{

FILE * fopen( const std::filesystem::path & filename, const char * mode )
{
#ifdef _WIN32
    std::wstring wmode( mode, mode + std::strlen( mode ) );
    return _wfopen( filename.wstring().c_str(), wmode.c_str() );
#else
    // Linux
    return std::fopen( utf8string( filename ).c_str(), mode );
#endif
}

FILE * File::open( const std::filesystem::path & filename, const char * mode ) 
{ 
    close();
    return handle_ = fopen( filename, mode ); 
}

void File::close()
{
    if ( !handle_ )
        return;
    std::fclose( handle_ );
    handle_ = nullptr;
}

} //namespace MR
