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
    enum class State
    {
        Init,
        TaskPreparing,
        TaskAwaiting,
        TaskProcessing,
        TaskFinished,
        Finished,
    };

    using Result = SceneLoad::SceneLoadResult;

    using Transition = StateMachine::Transition<State, Result>;
    using Executor = StateMachine::Executor<State, Result>;

    explicit AsyncSceneLoader( const std::vector<std::filesystem::path>& paths, ProgressCallback callback )
        : paths_( paths )
        , callback_( std::move( callback ) )
        , constructor_( std::make_shared<SceneConstructor>() )
    {
        //
    }

    Transition operator ()( State state )
    {
        switch ( state )
        {
            case State::Init:
                return init_();
            case State::TaskPreparing:
                return prepareTask_();
            case State::TaskAwaiting:
                return awaitForTask_();
            case State::TaskProcessing:
                return processTask_();
            case State::TaskFinished:
                return nextTask_();
            case State::Finished:
                return finished_();
        }
#ifdef __cpp_lib_unreachable
        std::unreachable();
#else
        assert( false );
        return {};
#endif
    }

private:
    Transition init_()
    {
        using namespace StateMachine;
        if ( paths_.empty() )
            return ContinueWith( State::Finished );
        else
            return ContinueWith( State::TaskPreparing );
    }

    Transition prepareTask_()
    {
        using namespace StateMachine;
        assert( currentPath_ < paths_.size() );
        const auto path = paths_[currentPath_];
        if ( path.empty() )
            return ContinueWith( State::TaskFinished );

        auto cb = [callback = callback_, index = currentPath_, number = paths_.size()] ( float v )
        {
            return callback( ( (float)index + v ) / (float)number );
        };

        if ( auto asyncFilter = findAsyncObjectLoadFilter( path ) )
        {
            spdlog::info( "Async loading file {}", utf8string( path ) );
            const auto asyncLoader = AsyncObjectLoad::getObjectLoader( *asyncFilter );
            assert( asyncLoader );
            // TODO: unify sync and async loader interfaces
            currentTask_ = asyncLoader( path, cb );
            return ContinueWith( State::TaskAwaiting );
        }
        else
        {
            spdlog::info( "Loading file {}", utf8string( path ) );
            currentResult_ = loadObjectFromFile( path, &warningTextBuffer_, cb );
            return ContinueWith( State::TaskProcessing );
        }
    }

    Transition awaitForTask_()
    {
        using namespace StateMachine;
        auto result = currentTask_();
        if ( !result )
            return Yield;
        currentResult_ = std::move( *result );
        return ContinueWith( State::TaskProcessing );
    }

    Transition processTask_()
    {
        using namespace StateMachine;
        const auto path = paths_[currentPath_];
        assert( !path.empty() );
        constructor_->process( path, std::move( currentResult_ ), std::move( warningTextBuffer_ ) );
        return ContinueWith( State::TaskFinished );
    }

    Transition nextTask_()
    {
        using namespace StateMachine;
        currentPath_ += 1;
        if ( currentPath_ >= paths_.size() )
            return ContinueWith( State::Finished );
        return ContinueWith( State::TaskPreparing );
    }

    Transition finished_()
    {
        using namespace StateMachine;
        return FinishWith( constructor_->construct() );
    }

private:
    std::vector<std::filesystem::path> paths_;
    ProgressCallback callback_;

    // TODO: make it copyable
    std::shared_ptr<SceneConstructor> constructor_;

    size_t currentPath_ { 0 };
    Resumable<Expected<std::vector<ObjectPtr>>> currentTask_;
    std::string warningTextBuffer_;
    Expected<std::vector<ObjectPtr>> currentResult_;
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

        std::string warningText;
        auto cb = [callback, index, count = files.size()] ( float v )
        {
            return callback( ( (float)index + v ) / (float)count );
        };
        spdlog::info( "Loading file {}", utf8string( path ) );
        auto res = loadObjectFromFile( path, &warningText, std::move( cb ) );
        constructor.process( path, std::move( res ), std::move( warningText ) );
    }

    return constructor.construct();
}

Resumable<SceneLoadResult>
asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    return AsyncSceneLoader::Executor( AsyncSceneLoader( files, std::move( callback ) ) );
}

} // namespace MR::SceneLoad
