#include "MRRasterLoad.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"

namespace MR
{

namespace RasterLoad
{

static RasterLoader findLoader( const std::filesystem::path& path )
{
    auto ext = toLower( utf8string( path.extension() ) );
    ext.insert( std::begin( ext ), '*' );
    return getRasterLoader( ext );
}

Expected<Raster> fromAnySupportedFormat( const std::filesystem::path& path, const RasterLoadSettings& settings )
{
    const auto loader = findLoader( path );
    if ( !loader.fileLoad )
        return unexpectedUnsupportedFileExtension();

    return loader.fileLoad( path, settings );
}

Expected<RasterInfo> infoFromAnySupportedFormat( const std::filesystem::path& path )
{
    const auto loader = findLoader( path );
    if ( !loader.infoLoad )
        return unexpectedUnsupportedFileExtension();

    return loader.infoLoad( path );
}

} // namespace RasterLoad

} // namespace MR
