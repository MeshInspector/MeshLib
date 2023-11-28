#include "MRSceneLoad.h"
#include "MRIOFormatsRegistry.h"
#include "MRObjectLoad.h"
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
    // TODO: resolve GCC bug
    //auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
    auto ext = utf8string( path.extension().u8string() );
    ext = std::string( "*" ) + ext;
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

struct AsyncLoadContext
{
    std::vector<std::filesystem::path> paths;
    std::vector<std::string> warningTexts;
    std::vector<Expected<std::vector<ObjectPtr>>> results;

    std::atomic_size_t asyncLoaderCount{ 0 };

    ProgressCallback progressCallback;
    std::map<size_t, float> asyncProgressMap;
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
    std::mutex asyncProgressMutex;
#endif

    void initializeProgressMap( const BitSet& asyncBitSet )
    {
        for ( const auto index : asyncBitSet )
            asyncProgressMap.emplace( index, .00f );
    }

    ProgressCallback progressCallbackFor( size_t index )
    {
        if ( !progressCallback )
            return {};

        return [this, index] ( float v )
        {
            float sum = .00f;
            {
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
                std::unique_lock lock( asyncProgressMutex );
#endif
                asyncProgressMap[index] = v;
                for ( const auto& [_, v1] : asyncProgressMap )
                    sum += v1 / (float)asyncProgressMap.size();
            }
            return reportProgress( progressCallback, sum );
        };
    }
};

}

namespace MR::SceneLoad
{

SceneLoadResult fromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
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

void asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files,
                                  SceneLoad::PostLoadCallback postLoadCallback, ProgressCallback progressCallback )
{
    auto ctx = std::make_shared<AsyncLoadContext>();
    ctx->paths = files;
    std::erase_if( ctx->paths, [] ( auto&& path ) { return path.empty(); } );

    const auto count = ctx->paths.size();
    ctx->warningTexts.resize( count );
    ctx->results.resize( count, unexpected( "Uninitialized" ) );

    BitSet toAsyncLoad( count );
    size_t syncIndex = 0;
    for ( auto index = 0ull; index < count; ++index )
    {
        const auto& path = ctx->paths[index];
        if ( findAsyncObjectLoadFilter( path ) )
        {
            toAsyncLoad.set( index );
        }
        else
        {
            spdlog::info( "Loading file {}", utf8string( path ) );
            ctx->results[index] = loadObjectFromFile( path, &ctx->warningTexts[index], subprogress( progressCallback, syncIndex++, count ) );
        }
    }
    assert( syncIndex + toAsyncLoad.count() == count );

    ctx->progressCallback = subprogress( progressCallback, (float)syncIndex / (float)count, 1.00f );
    ctx->initializeProgressMap( toAsyncLoad );

    auto postLoad = [ctx, count, postLoadCallback]
    {
        SceneConstructor constructor;
        for ( auto index = 0ull; index < count; ++index )
            constructor.process( ctx->paths[index], ctx->results[index], ctx->warningTexts[index] );
        postLoadCallback( constructor.construct() );
    };

    if ( toAsyncLoad.none() )
        return postLoad();

    ctx->asyncLoaderCount = toAsyncLoad.count();
    for ( const auto index : toAsyncLoad )
    {
        const auto& path = ctx->paths[index];
        const auto asyncFilter = findAsyncObjectLoadFilter( path );
        assert( asyncFilter );
        const auto asyncLoader = AsyncObjectLoad::getObjectLoader( *asyncFilter );
        assert( asyncLoader );
        const auto callback = ctx->progressCallbackFor( index );
        spdlog::info( "Async loading file {}", utf8string( path ) );
        // TODO: unify sync and async loader interfaces
        asyncLoader( path, [ctx, index, postLoad, callback] ( Expected<std::vector<ObjectPtr>> result )
        {
            ctx->results[index] = std::move( result );
            reportProgress( callback, 1.00f );
            if ( ctx->asyncLoaderCount.fetch_sub( 1 ) == 1 )
                // that was the last file
                postLoad();
        }, callback );
    }
}

} // namespace MR::SceneLoad
