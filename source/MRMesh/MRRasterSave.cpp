#include "MRRasterSave.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"

namespace MR
{

namespace RasterSave
{

Expected<void> toAnySupportedFormat( const Raster& raster, const std::filesystem::path& path, const RasterSaveSettings& settings )
{
    auto ext = toLower( utf8string( path.extension() ) );
    ext.insert( std::begin( ext ), '*' );

    auto saver = getRasterSaver( ext );
    if ( !saver )
        return unexpectedUnsupportedFileExtension();

    return saver( raster, path, settings );
}

} // namespace RasterSave

} // namespace MR
