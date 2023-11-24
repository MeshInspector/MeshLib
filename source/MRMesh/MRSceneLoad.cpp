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

class AsyncSceneLoader : public ResumableTask<SceneLoad::SceneLoadResult>
{
public:
    explicit AsyncSceneLoader( const std::vector<std::filesystem::path>& paths, ProgressCallback callback )
        : paths_( paths )
        , callback_ ( std::move( callback ) )
        , state_( State::Init )
    {
        //
    }
    ~AsyncSceneLoader() override = default;

    void start() override
    {
        if ( state_ != State::Init )
            return;

        if ( paths_.empty() )
            state_ = State::Finished;
        else
            state_ = State::TaskPreparing;
    }

    bool resume() override
    {
        while ( !process_() );
        return state_ == State::Finished;
    }

    SceneLoad::SceneLoadResult result() const override
    {
        return result_;
    }

private:
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

    bool process_()
    {
        switch ( state_ )
        {
            case State::Init:
                return true;
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
                return true;
        }
#ifdef __cpp_lib_unreachable
        std::unreachable();
#else
        assert( false );
        return {};
#endif
    }

    bool continueWith_( State state )
    {
        state_ = state;
        return false;
    }

    static bool suspend_()
    {
        return true;
    }

    bool prepareTask_()
    {
        assert( state_ == State::TaskPreparing );
        assert( currentPath_ < paths_.size() );
        const auto path = paths_[currentPath_];
        if ( path.empty() )
            return continueWith_( State::TaskFinished );

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
            return continueWith_( State::TaskAwaiting );
        }

        spdlog::info( "Loading file {}", fileName );
        currentResult_ = loadObjectFromFile( path, &warningTextBuffer_, cb );
        return continueWith_( State::TaskProcessing );
    }

    bool awaitForTask_()
    {
        assert( state_ == State::TaskAwaiting );
        auto result = currentTask_();
        if ( !result )
            return suspend_();
        currentResult_ = std::move( *result );
        return continueWith_( State::TaskProcessing );
    }

    bool processTask_()
    {
        assert( state_ == State::TaskProcessing );
        const auto path = paths_[currentPath_];
        assert( !path.empty() );

        const auto fileName = utf8string( path );
        spdlog::info( "Load file {} - {}", fileName, currentResult_.has_value() ? "success" : currentResult_.error().c_str() );
        if ( !currentResult_.has_value() )
        {
            // TODO: user-defined error format
            errorSummary_ << ( !isEmpty( errorSummary_ ) ? "\n" : "" ) << "\n" << currentResult_.error();
            return continueWith_( State::TaskFinished );
        }

        if ( !warningTextBuffer_.empty() )
        {
            // TODO: user-defined warning format
            warningSummary_ << ( !isEmpty( warningSummary_ ) ? "\n" : "" ) << "\n" << fileName << ":\n" << warningTextBuffer_ << "\n";
        }

        const auto prevObjectCount = loadedObjects_.size();
        for ( auto& obj : *currentResult_ )
            if ( obj )
                loadedObjects_.emplace_back( std::move( obj ) );
        if ( prevObjectCount != loadedObjects_.size() )
            loadedFiles_.emplace_back( path );
        else
            errorSummary_ << ( !isEmpty( errorSummary_ ) ? "\n" : "" ) << "\n" << "No objects found in the file \"" << fileName << "\"";

        return continueWith_( State::TaskFinished );
    }

    bool nextTask_()
    {
        assert( state_ == State::TaskFinished );
        currentPath_ += 1;
        if ( currentPath_ >= paths_.size() )
            return continueWith_( State::Finishing );
        return continueWith_( State::TaskPreparing );
    }

    bool finishing_()
    {
        assert( state_ == State::Finishing );
        result_ = construct( std::move( loadedFiles_ ), std::move( loadedObjects_ ), errorSummary_.str(), warningSummary_.str() );
        state_ = State::Finished;
        return true;
    }

private:
    std::vector<std::filesystem::path> paths_;
    ProgressCallback callback_;

    SceneLoad::SceneLoadResult result_;
    std::vector<std::filesystem::path> loadedFiles_;
    std::vector<std::shared_ptr<Object>> loadedObjects_;
    std::ostringstream errorSummary_;
    std::ostringstream warningSummary_;

    State state_;
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

std::shared_ptr<ResumableTask<SceneLoadResult>>
asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    return std::make_shared<AsyncSceneLoader>( files, std::move( callback ) );
}

} // namespace MR::SceneLoad
