#include "MRGcodeSave.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include <fstream>

namespace MR
{

namespace GcodeSave
{

const IOFilters Filters =
{
    {"G-code (.gcode)", "*.gcode"},
    {"Numerical Control (.nc)", "*.nc"},
    {"Text file (.txt)", "*.txt"}
};

Expected<void> toGcode( const GcodeSource& gcode, const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toGcode( gcode, out, callback );
}

Expected<void> toGcode( const GcodeSource& gcode, std::ostream& out, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;
    for ( size_t i = 0; i < gcode.size(); ++i )
    {
        // lines from GcodeLoad have no line ends, while exportToolPathToGCode keeps them
        const auto& s = gcode[i];
        out << s;
        if ( s.empty() || s.back() != '\n' )
            out << '\n';
        if ( !reportProgress( callback, [&] { return float( i ) / gcode.size(); }, i, 1024 ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving G-code" ) );

    reportProgress( callback, 1.f );
    return {};
}

Expected<void> toAnySupportedFormat( const GcodeSource& gcode, const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    auto ext = toLower( utf8string( file.extension() ) );
    if ( ext == ".gcode" || ext == ".txt" || ext == ".nc" )
        return toGcode( gcode, file, callback );
    return unexpectedUnsupportedFileExtension();
}

Expected<void> toAnySupportedFormat( const GcodeSource& gcode, std::ostream& out, const std::string& extension, ProgressCallback callback /*= {} */ )
{
    auto ext = toLower( extension );
    if ( ext == "*.gcode" || ext == "*.txt" || ext == "*.nc" )
        return toGcode( gcode, out, callback );
    return unexpectedUnsupportedFileExtension();
}

}

}
