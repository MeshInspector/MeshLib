#include "MRGcodeLoad.h"
#include "MRStringConvert.h"
#include <fstream>
#include <sstream>

namespace MR
{

namespace GcodeLoad
{

const IOFilters Filters =
{
    {"G-code", "*.gcode"},
    {"Numerical Control", "*.nc"},
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

Expected<GcodeSource, std::string> fromGcode( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    std::ifstream filestream( file );
    return fromGcode( filestream, callback );
}

Expected<MR::GcodeSource, std::string> fromGcode( std::istream& in, ProgressCallback callback /*= {} */ )
{
    std::stringstream buffer;
    buffer << in.rdbuf();

    return splitString( buffer.str() );
}

Expected<GcodeSource, std::string> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<std::vector<std::string>, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".gcode" || ext == ".txt" || ext == ".nc" )
        res = fromGcode( file, callback );
    return res;
}

Expected<MR::GcodeSource, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, ProgressCallback callback /*= {} */ )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<GcodeSource, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".gcode" || ext == ".txt" || ext == ".nc" )
        res = fromGcode( in, callback );
    return res;
}

}

}
