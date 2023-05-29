#include "MRGcodeLoad.h"
#include "MRStringConvert.h"

namespace MR
{

namespace GcodeLoad
{

const IOFilters Filters =
{
    {"G-code", "*.gcode"},
    {"Text file", "*.txt"}
};

GcodeSource splitString( const std::string& source )
{
    std::vector<std::string> res;
    size_t frameBegin = 0;
    size_t frameEnd = 0;
    frameEnd = source.find( '\n', frameBegin );
    while ( frameEnd != std::string::npos )
    {
        res.push_back( std::string( source.begin() + frameBegin, source.begin() + frameEnd ) );
        frameBegin = frameEnd + 1;
        frameEnd = source.find( '\n', frameBegin );
    }
    res.push_back( std::string( source.begin() + frameBegin, source.end() ) );
    return res;
}

tl::expected<GcodeSource, std::string> fromGcode( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    std::ifstream filestream( file );
    std::stringstream buffer;
    buffer << filestream.rdbuf();

    return splitString( buffer.str() );
}

tl::expected<GcodeSource, std::string> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<std::vector<std::string>, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".gcode" || ext == ".txt" )
        res = fromGcode( file, callback );
    return res;
}

}

}
