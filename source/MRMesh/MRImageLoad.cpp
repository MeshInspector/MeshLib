#include "MRImageLoad.h"
#include "MRIOFormatsRegistry.h"
#include "MRImage.h"
#include "MRStringConvert.h"
#include "MRExpected.h"

#include <filesystem>
#include <string>

namespace MR
{
namespace ImageLoad
{

Expected<Image> fromAnySupportedFormat( const std::filesystem::path& file )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );
    ext = "*" + ext;

    auto loader = getImageLoader( ext );
    if ( !loader )
        return unexpected( std::string( "unsupported file extension" ) );

    return loader( file );
}

}

}
