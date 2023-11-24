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

SceneLoad::SceneLoadResult construct( std::vector<std::filesystem::path> loadedFiles,
                                      std::vector<std::shared_ptr<Object>> loadedObjects,
                                      std::string errorSummary, std::string warningSummary )
{
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
        .errorSummary = std::move( errorSummary ),
        .warningSummary = std::move( warningSummary ),
    };
}

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
        Finishing,
        Finished,
    };

    using Result = SceneLoad::SceneLoadResult;

    using Transition = StateMachine::Transition<State, Result>;
    using Executor = StateMachine::Executor<State, Result>;

    explicit AsyncSceneLoader( const std::vector<std::filesystem::path>& paths, ProgressCallback callback )
        : paths_( paths )
        , callback_ ( std::move( callback ) )
        , errorSummary_( std::make_shared<std::ostringstream>() )
        , warningSummary_( std::make_shared<std::ostringstream>() )
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
            case State::Finishing:
                return finishing_();
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

        const auto fileName = utf8string( path );

        auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
        for ( auto& c : ext )
            c = (char)std::tolower( c );

        auto cb = [callback = callback_, index = currentPath_, number = paths_.size()] ( float v )
        {
            return callback( ( (float)index + v ) / (float)number );
        };

        const auto asyncFilters = AsyncObjectLoad::getFilters();
        const auto asyncFilter = std::find_if( asyncFilters.begin(), asyncFilters.end(), [&ext] ( auto&& filter )
        {
            return filter.extensions.find( ext ) != std::string::npos;
        } );
        if ( asyncFilter != asyncFilters.end() )
        {
            const auto asyncLoader = AsyncObjectLoad::getObjectLoader( *asyncFilter );
            assert( asyncLoader );
            spdlog::info( "Async loading file {}", fileName );
            currentTask_ = asyncLoader( path, cb );
            return ContinueWith( State::TaskAwaiting );
        }

        spdlog::info( "Loading file {}", fileName );
        currentResult_ = loadObjectFromFile( path, &warningTextBuffer_, cb );
        return ContinueWith( State::TaskProcessing );
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

        const auto fileName = utf8string( path );
        spdlog::info( "Load file {} - {}", fileName, currentResult_.has_value() ? "success" : currentResult_.error().c_str() );
        if ( !currentResult_.has_value() )
        {
            // TODO: user-defined error format
            *errorSummary_ << ( !isEmpty( *errorSummary_ ) ? "\n" : "" ) << "\n" << currentResult_.error();
            return ContinueWith( State::TaskFinished );
        }

        if ( !warningTextBuffer_.empty() )
        {
            // TODO: user-defined warning format
            *warningSummary_ << ( !isEmpty( *warningSummary_ ) ? "\n" : "" ) << "\n" << fileName << ":\n" << warningTextBuffer_ << "\n";
        }

        const auto prevObjectCount = loadedObjects_.size();
        for ( auto& obj : *currentResult_ )
            if ( obj )
                loadedObjects_.emplace_back( std::move( obj ) );
        if ( prevObjectCount != loadedObjects_.size() )
            loadedFiles_.emplace_back( path );
        else
            *errorSummary_ << ( !isEmpty( *errorSummary_ ) ? "\n" : "" ) << "\n" << "No objects found in the file \"" << fileName << "\"";

        return ContinueWith( State::TaskFinished );
    }

    Transition nextTask_()
    {
        using namespace StateMachine;
        currentPath_ += 1;
        if ( currentPath_ >= paths_.size() )
            return ContinueWith( State::Finishing );
        return ContinueWith( State::TaskPreparing );
    }

    Transition finishing_()
    {
        using namespace StateMachine;
        result_ = construct( std::move( loadedFiles_ ), std::move( loadedObjects_ ), errorSummary_->str(), warningSummary_->str() );
        return ContinueWith( State::Finished );
    }

    Transition finished_()
    {
        using namespace StateMachine;
        return FinishWith( Result { result_ } );
    }

private:
    std::vector<std::filesystem::path> paths_;
    ProgressCallback callback_;

    SceneLoad::SceneLoadResult result_;
    std::vector<std::filesystem::path> loadedFiles_;
    std::vector<std::shared_ptr<Object>> loadedObjects_;
    // TODO: make them copyable
    std::shared_ptr<std::ostringstream> errorSummary_;
    std::shared_ptr<std::ostringstream> warningSummary_;

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

    return construct( std::move( loadedFiles ), std::move( loadedObjects ), errorSummary.str(), warningSummary.str() );
}

Resumable<SceneLoadResult>
asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    return AsyncSceneLoader::Executor( AsyncSceneLoader( files, std::move( callback ) ) );
}

} // namespace MR::SceneLoad
