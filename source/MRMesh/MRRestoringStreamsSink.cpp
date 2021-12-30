#include "MRRestoringStreamsSink.h"
#include <iostream>

namespace MR
{

LoggingStreambuf::LoggingStreambuf( spdlog::level::level_enum level )
    : level_( level )
{
}

std::streamsize LoggingStreambuf::xsputn( const char_type* s, std::streamsize count )
{
    auto remaining = count;
    std::unique_lock lock( bufMutex_ );
    while ( remaining-- > 0 )
    {
        auto ch = *s++;
        if ( ch == char_type{'\n'} )
        {
            spdlog::log( level_, buf_ );
            buf_.clear();
        }
        else
        {
            buf_ += ch;
        }
    }
    return count;
}

LoggingStreambuf::int_type LoggingStreambuf::overflow( int_type ch )
{
    if ( ch != traits_type::eof() )
    {
        auto value = static_cast<char_type>( ch );
        return int_type( xsputn( &value, 1 ) );
    }
    else
    {
        return 0;
    }
}

RestoringStreamsSink::RestoringStreamsSink() :
    spdCoutBuf_( spdlog::level::info ),
    spdCerrBuf_( spdlog::level::err ),
    spdClogBuf_( spdlog::level::trace )
{
    coutBuf_ = std::cout.rdbuf();
    cerrBuf_ = std::cerr.rdbuf();
    clogBuf_ = std::clog.rdbuf();

    std::cout.rdbuf( &spdCoutBuf_ );
    std::cerr.rdbuf( &spdCerrBuf_ );
    std::clog.rdbuf( &spdClogBuf_ );
}

RestoringStreamsSink::~RestoringStreamsSink()
{
    std::cout.rdbuf( coutBuf_ );
    std::cerr.rdbuf( cerrBuf_ );
    std::clog.rdbuf( clogBuf_ );
}

} //namespace MR
