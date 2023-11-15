#include "MRSceneLoad.h"
#include "MRObjectLoad.h"
#include "MRStringConvert.h"

#include <MRPch/MRSpdlog.h>

namespace
{

inline bool isEmpty( std::ostream& os )
{
    return os.tellp() == std::streampos( 0 );
}

}

namespace MR::SceneLoad
{

SceneLoadResult
fromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    std::vector<std::filesystem::path> loadedFiles;
    std::vector<std::shared_ptr<Object>> loadedObjects;
    std::ostringstream errorSummary;
    std::ostringstream warningSummary;

    for ( auto index = 0ull; index < files.size(); ++index )
    {
        const auto& path = files[index];
        if ( path.empty() )
            continue;

        const auto fileName = utf8string( path );
        spdlog::info( "Loading file {}", fileName );
        std::string warningText;
        auto res = loadObjectFromFile( path, &warningText, [callback, index, count = files.size()] ( float v )
        {
            return callback( ( (float)index + v ) / (float)count );
        } );
        spdlog::info( "Load file {} - {}", fileName, res.has_value() ? "success" : res.error().c_str() );
        if ( !res.has_value() )
        {
            // TODO: user-defined error format
            errorSummary << ( !isEmpty( errorSummary ) ? "\n" : "" ) << "\n" << res.error();
            continue;
        }
        if ( !warningText.empty() )
        {
            // TODO: user-defined warning format
            warningSummary << ( !isEmpty( warningSummary ) ? "\n" : "" ) << "\n" << fileName << ":\n" << warningText << "\n";
        }

        const auto prevObjectCount = loadedObjects.size();
        for ( auto& obj : *res )
            if ( obj )
                loadedObjects.emplace_back( std::move( obj ) );
        if ( prevObjectCount != loadedObjects.size() )
            loadedFiles.emplace_back( path );
        else
            errorSummary << ( !isEmpty( errorSummary ) ? "\n" : "" ) << "\n" << "No objects found in the file \"" << fileName << "\"";
    }

    auto scene = std::make_shared<Object>();
    bool constructed;
    if ( loadedObjects.size() == 1 )
    {
        auto& object = loadedObjects.front();
        if ( object->typeName() == Object::TypeName() )
        {
            constructed = false;
            scene = std::move( object );
        }
        else
        {
            constructed = true;
            scene->addChild( std::move( object ) );
        }
    }
    else
    {
        constructed = true;
        for ( auto& object : loadedObjects )
            scene->addChild( std::move( object ) );
    }

    return {
        .scene = std::move( scene ),
        .isSceneConstructed = constructed,
        .loadedFiles = std::move( loadedFiles ),
        .errorSummary = errorSummary.str(),
        .warningSummary = warningSummary.str(),
    };
}

std::shared_ptr<ResumableTask<SceneLoadResult>>
asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    return {};
}

} // namespace MR::SceneLoad
