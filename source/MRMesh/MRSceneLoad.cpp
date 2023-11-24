#include "MRSceneLoad.h"
#include "MRIOFormatsRegistry.h"
#include "MRObjectLoad.h"
#include "MRStateMachine.h"
#include "MRStringConvert.h"

#include <MRPch/MRSpdlog.h>

namespace
{

using namespace MR;

inline bool isEmpty( std::ostream& os )
{
    return os.tellp() == std::streampos( 0 );
}

std::optional<IOFilter> findAsyncObjectLoadFilter( const std::filesystem::path& path )
{
    auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
    for ( auto& c : ext )
        c = (char)std::tolower( c );

    const auto asyncFilters = AsyncObjectLoad::getFilters();
    const auto asyncFilter = std::find_if( asyncFilters.begin(), asyncFilters.end(), [&ext] ( auto&& filter )
    {
        return filter.extensions.find( ext ) != std::string::npos;
    } );
    if ( asyncFilter != asyncFilters.end() )
        return *asyncFilter;
    else
        return std::nullopt;
}

using Result = Expected<std::vector<ObjectPtr>>;

Resumable<Result> getLoader( const std::filesystem::path& path, std::string* loadWarn, ProgressCallback callback )
{
    assert( !path.empty() );
    if ( auto asyncFilter = findAsyncObjectLoadFilter( path ) )
    {
        spdlog::info( "Async loading file {}", utf8string( path ) );
        const auto asyncLoader = AsyncObjectLoad::getObjectLoader( *asyncFilter );
        assert( asyncLoader );
        // TODO: unify sync and async loader interfaces
        return asyncLoader( path, std::move( callback ) );
    }
    else
    {
        spdlog::info( "Loading file {}", utf8string( path ) );
        return [path, loadWarn, callback] () -> std::optional<Result>
        {
            return loadObjectFromFile( path, loadWarn, callback );
        };
    }
}

class SceneConstructor
{
public:
    void process( const std::filesystem::path& path, Expected<std::vector<ObjectPtr>> res, std::string warningText )
    {
        const auto fileName = utf8string( path );
        spdlog::info( "Load file {} - {}", fileName, res.has_value() ? "success" : res.error().c_str() );
        if ( !res.has_value() )
        {
            // TODO: user-defined error format
            errorSummary_ << ( !isEmpty( errorSummary_ ) ? "\n" : "" ) << "\n" << res.error();
            return;
        }
        if ( !warningText.empty() )
        {
            // TODO: user-defined warning format
            warningSummary_ << ( !isEmpty( warningSummary_ ) ? "\n" : "" ) << "\n" << fileName << ":\n" << warningText << "\n";
        }

        const auto prevObjectCount = loadedObjects_.size();
        for ( auto& obj : *res )
            if ( obj )
                loadedObjects_.emplace_back( std::move( obj ) );
        if ( prevObjectCount != loadedObjects_.size() )
            loadedFiles_.emplace_back( path );
        else
            errorSummary_ << ( !isEmpty( errorSummary_ ) ? "\n" : "" ) << "\n" << "No objects found in the file \"" << fileName << "\"";
    }

    SceneLoad::SceneLoadResult construct() const
    {
        auto scene = std::make_shared<Object>();
        bool constructed;
        if ( loadedObjects_.size() == 1 )
        {
            const auto& object = loadedObjects_.front();
            if ( object->typeName() == Object::TypeName() )
            {
                constructed = false;
                scene = object;
            }
            else
            {
                constructed = true;
                scene->addChild( object );
            }
        }
        else
        {
            constructed = true;
            for ( const auto& object : loadedObjects_ )
                scene->addChild( object );
        }

        return {
            .scene = std::move( scene ),
            .isSceneConstructed = constructed,
            .loadedFiles = loadedFiles_,
            .errorSummary = errorSummary_.str(),
            .warningSummary = warningSummary_.str(),
        };
    }

private:
    std::vector<std::filesystem::path> loadedFiles_;
    std::vector<std::shared_ptr<Object>> loadedObjects_;
    std::ostringstream errorSummary_;
    std::ostringstream warningSummary_;
};

class AsyncSceneLoader
{
public:
    explicit AsyncSceneLoader( const std::vector<std::filesystem::path>& paths, ProgressCallback callback )
        : paths_( paths )
        , callback_( std::move( callback ) )
    {
        std::erase_if( paths_, [] ( auto&& path ) { return path.empty(); } );
    }

    std::optional<SceneLoad::SceneLoadResult> operator ()()
    {
        if ( paths_.empty() )
            return SceneLoad::SceneLoadResult();

        // initialize loaders
        if ( loaders_.empty() )
        {
            warningTexts_.resize( paths_.size() );
            loaders_.resize( paths_.size() );
            for ( auto index = 0ull; index < paths_.size(); ++index )
            {
                assert( !paths_[index].empty() );
                loaders_[index] = cached( getLoader( paths_[index], &warningTexts_[index], subprogress( callback_, index, paths_.size() ) ) );
            }
        }

        // waiting for all loaders to finish
        for ( auto& loader : loaders_ )
            if ( !loader() )
                return std::nullopt;

        // gather loaded objects
        SceneConstructor constructor;
        for ( auto index = 0ull; index < paths_.size(); ++index )
        {
            auto result = loaders_[index]();
            assert( result );
            constructor.process( paths_[index], std::move( *result ), warningTexts_[index] );
        }
        return constructor.construct();
    }

private:
    std::vector<std::filesystem::path> paths_;
    ProgressCallback callback_;

    std::vector<std::string> warningTexts_;
    std::vector<Resumable<Result>> loaders_;
};

}

namespace MR::SceneLoad
{

SceneLoadResult
fromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    SceneConstructor constructor;
    for ( auto index = 0ull; index < files.size(); ++index )
    {
        const auto& path = files[index];
        if ( path.empty() )
            continue;

        spdlog::info( "Loading file {}", utf8string( path ) );
        std::string warningText;
        auto result = loadObjectFromFile( path, &warningText, subprogress( callback, index, files.size() ) );
        constructor.process( path, std::move( result ), std::move( warningText ) );
    }
    return constructor.construct();
}

Resumable<SceneLoadResult>
asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    return AsyncSceneLoader( files, std::move( callback ) );
}

} // namespace MR::SceneLoad
