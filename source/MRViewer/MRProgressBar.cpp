#include "MRProgressBar.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "ImGuiHelpers.h"
#include "MRFrameRedrawRequest.h"
#include "MRImGui.h"
#include "MRRibbonButtonDrawer.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRTimeRecord.h"
#include "MRRibbonConstants.h"
#include "MRUIStyle.h"
#include "MRCommandLoop.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonMenu.h"
#include "MRViewer/MRUITestEngine.h"
#include "imgui_internal.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include <boost/exception/diagnostic_information.hpp>
#include <GLFW/glfw3.h>
#include <atomic>
#include <thread>

#ifdef _WIN32
#include <excpt.h>
#endif
#if defined( __EMSCRIPTEN__ )
#if  !defined( __EMSCRIPTEN_PTHREADS__ )
namespace
{
std::function<void()> staticTaskForLaterCall;
void asyncCallTask( void * )
{
    if ( staticTaskForLaterCall )
        staticTaskForLaterCall();
    staticTaskForLaterCall = {};
}
}
#endif
extern "C" {

EMSCRIPTEN_KEEPALIVE bool emsIsProgressBarOrdered()
{
    return MR::ProgressBar::isOrdered();
}
}
#endif

namespace MR
{

namespace
{

class ProgressBarImpl
{
public:
    static ProgressBarImpl& instance();

    ProgressBarImpl();
    ~ProgressBarImpl();

    void initialize_( std::string title, int taskCount, std::function<void()> postInit);

    // cover task execution with try catch block (in release only)
    // if catches exception shows error in main thread overriding user defined main thread post-processing
    [[maybe_unused]] bool tryRun_( const std::function<bool ()>& task );
    bool tryRunWithSehHandler_( const std::function<bool ()>& task );

    float lastOperationTimeSec_{ -1.0f };
    Time operationStartTime_;
    std::atomic<float> progress_;
    std::atomic<int> currentTask_, taskCount_;
    std::mutex mutex_;
    std::string taskName_, title_;
    bool overrideTaskName_{ false };

    FrameRedrawRequest frameRequest_;

    // parameter is needed for logging progress
    std::atomic<int> percents_;

    std::thread thread_;
    std::function<void()> onFinish_;

    std::atomic<bool> allowCancel_;
    std::atomic<bool> canceled_;
    std::atomic<bool> finished_;
    ImGuiID setupId_ = ImGuiID( -1 );

    bool isOrdered_{ false };

    // needed to be able to call progress bar from any point, not only from ImGui frame scope
    bool deferredOpenPopup_{ false };

    // this is needed to show full progress before closing
    bool closeDialogNextFrame_{ false };

    ThreadRootTimeRecord rootTimeRecord_{ "Progress" };
};

ProgressBarImpl& ProgressBarImpl::instance()
{
    static ProgressBarImpl inst;
    return inst;
}

ProgressBarImpl::ProgressBarImpl() :
    progress_( 0.0f ), currentTask_( 0 ), taskCount_( 1 ),
    taskName_( "Current task" ), title_( "Sample Title" ),
    canceled_{ false }, finished_{ false }
{}

ProgressBarImpl::~ProgressBarImpl()
{
    canceled_ = true;
    if ( thread_.joinable() )
        thread_.join();
}

void ProgressBarImpl::initialize_( std::string title, int taskCount, std::function<void()> postInit)
{
    if ( finished_ && thread_.joinable() )
        thread_.join();

    deferredOpenPopup_ = true;

    progress_ = 0.0f;

    taskCount_ = taskCount;
    currentTask_ = 0;
    if ( taskCount_ == 1 )
        currentTask_ = 1;

    closeDialogNextFrame_ = false;
    canceled_ = false;
    finished_ = false;

    {
        std::unique_lock lock( mutex_ );
        title_ = std::move( title );
    }

    frameRequest_.reset();    
    operationStartTime_ = std::chrono::system_clock::now();
    if ( postInit )
        postInit();
}

bool ProgressBarImpl::tryRun_( const std::function<bool ()>& task )
{
#ifndef NDEBUG
    return task();
#else
    try
    {
        return task();
    }
    catch ( const std::bad_alloc& badAllocE )
    {
        onFinish_ = [msg = std::string( badAllocE.what() )]
        {
            spdlog::error( msg );
            showError( "Not enough memory for the requested operation." );
        };
        return true;
    }
    catch ( ... )
    {
        onFinish_ = [msg = boost::current_exception_diagnostic_information()]
        {
            showError( msg );
        };
        return true;
    }
#endif
}

bool ProgressBarImpl::tryRunWithSehHandler_( const std::function<bool()>& task )
{
#ifndef _WIN32
    return tryRun_( task );
#else
#ifndef NDEBUG
    return task();
#else
    __try
    {
        return tryRun_( task );
    }
    __except ( EXCEPTION_EXECUTE_HANDLER )
    {
        onFinish_ = []
        {
            showError( "Unknown exception occurred" );
        };
        return true;
    }
#endif
#endif
}

} //anonymous namespace

namespace ProgressBar
{

void setup( float scaling )
{
    auto& instance = ProgressBarImpl::ProgressBarImpl::instance();

    if ( instance.deferredOpenPopup_ && instance.setupId_ != ImGuiID( -1 ) )
    {
        instance.deferredOpenPopup_ = false;
        bool thisOpen = ImGui::IsPopupOpen( instance.setupId_, 0 );
        bool isAnyOpen = ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
        if ( isAnyOpen && !thisOpen )
            ImGui::CloseCurrentPopup();
        if ( !thisOpen )
            ImGui::OpenPopup( instance.setupId_ );
    }

    instance.setupId_ = ImGui::GetID( "###GlobalProgressBarPopup" );
    const Vector2f windowSize( 440.0f * scaling, 144.0f * scaling );
    auto& viewer = getViewerInstance();
    ImGui::SetNextWindowPos( 0.5f * ( Vector2f( viewer.framebufferSize ) - windowSize ), ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    if ( ImGui::BeginModalNoAnimation( "###GlobalProgressBarPopup", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        UI::TestEngine::pushTree( "ProgressBar" );
        MR_FINALLY{ UI::TestEngine::popTree(); };

        instance.frameRequest_.reset();

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
        auto smallFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Small );
        if ( smallFont )
            ImGui::PushFont( smallFont );
        ImGui::PushStyleColor( ImGuiCol_Text, StyleConsts::ProgressBar::textColor.getUInt32() );
        ImGui::SetCursorPos( ImVec2( 32.0f * scaling, 20.0f * scaling ) );
        {
            std::unique_lock lock( instance.mutex_ );
            if ( instance.overrideTaskName_ )
            {
                ImGui::Text( "%s : %s", instance.title_.c_str(), instance.taskName_.c_str() );
            }
            else if ( instance.taskCount_ > 1 )
            {
                ImGui::Text( "%s :", instance.title_.c_str() );
                ImGui::SameLine();
                ImGui::Text( "%s (%d/%d)\n", instance.taskName_.c_str(), ( int )instance.currentTask_, ( int )instance.taskCount_ );
            }
            else
            {
                ImGui::Text( "%s", instance.title_.c_str() );
            }
        }
        ImGui::PopStyleColor();
        if ( smallFont )
            ImGui::PopFont();

        auto progress = (float)instance.progress_;
        ImGui::SetCursorPos( ImVec2( 32.0f * scaling, 56.0f * scaling ) );
        UI::progressBar( scaling, progress, ImVec2( 380.0f * scaling, 12.0f * scaling ) );

        if ( instance.allowCancel_ )
        {
            ImVec2 btnSize = ImVec2( 90.0f * scaling, 28.0f * scaling );
            ImGui::SetCursorPos( ImVec2( ( windowSize.x - btnSize.x ) * 0.5f, 92.0f * scaling ) );
            if ( !instance.canceled_ )
            {
                if ( UI::button( "Cancel", btnSize, ImGuiKey_Escape ) )
                {
                    std::unique_lock lock( instance.mutex_ );
                    spdlog::info( "Operation progress: \"{}\" - Canceling", instance.title_ );
                    instance.canceled_ = true;
                }
            }
            else
            {
                ImGui::Text( "Canceling..." );
            }
        }
#else
        auto textSize = ImGui::CalcTextSize( "Operation is in progress, please wait..." );
        ImGui::SetCursorPos( 0.5f * ( windowSize - Vector2f( textSize ) ) );
        ImGui::Text( "Operation is in progress, please wait..." );
#endif
        if ( instance.closeDialogNextFrame_ )
        {
            instance.closeDialogNextFrame_ = false;
            ImGui::CloseCurrentPopup();
            getViewerInstance().incrementForceRedrawFrames( 1, true );
        }
        if ( instance.finished_ )
        {
            if ( instance.isOrdered_ )
            {
                const float time = float( ( std::chrono::duration_cast< std::chrono::milliseconds >( std::chrono::system_clock::now() - instance.operationStartTime_ ) ).count() ) * 1e-3f;
                instance.lastOperationTimeSec_ = time;
                spdlog::info( "Operation \"{}\" time  - {} sec", instance.title_, instance.lastOperationTimeSec_ );
                pushNotification( { .header = fmt::format( "{:.2f} sec", time < 5.e-3f ? 0.f : time ),
                                    .text = instance.title_, .type = NotificationType::Time,.tags = NotificationTags::Report } );
            }
            instance.isOrdered_ = false;
            instance.closeDialogNextFrame_ = true;
            // important to be after `isOrdered_=false` and `closeDialogNextFrame_=true`
            // to handle progress bar ordered in `onFinish_`
            if ( instance.onFinish_ )
            {
                instance.onFinish_();
                instance.onFinish_ = {};
            }
            getViewerInstance().incrementForceRedrawFrames();
        }
        ImGui::EndPopup();
    }
}

void onFrameEnd()
{
    // this is needed to prevent unexpected closing on progress bar window in:
    // ImGui::NewFrame() / ImGui::UpdateMouseMovingWindowNewFrame() / ImGui::FocusWindow()
    // that can happen if progress bar is ordered on clicking to the window
    // (for example on finish editing some InputFloat, clicking on window makes ImGui think this window is moving
    //  and close progress bar modal before it starts, task of progress bar is going but post-processing is not)
    auto& inst = ProgressBarImpl::instance();
    if ( !inst.isOrdered_ )
        return;
    auto ctx = ImGui::GetCurrentContext();
    if ( !ctx )
        return;
    if ( !ctx->MovingWindow )
        return;
    if ( std::string( ctx->MovingWindow->Name ).ends_with( "###GlobalProgressBarPopup" ) )
        return;
    ctx->MovingWindow = nullptr;
}

void order( const char* name, const std::function<void()>& task, int taskCount )
{
    return orderWithMainThreadPostProcessing(
        name,
        [task] ()
    {
        task(); return [] ()
        {};
    },
        taskCount );
}

void orderWithMainThreadPostProcessing( const char* name, TaskWithMainThreadPostProcessing task, int taskCount )
{
    auto& instance = ProgressBarImpl::instance();

    if ( isFinished() && instance.thread_.joinable() )
        instance.thread_.join();

    instance.isOrdered_ = true;

    auto postInit = [&instance, task]
    {
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
        instance.thread_ = std::thread( [&instance, task]
        {
            registerThreadRootTimeRecord( instance.rootTimeRecord_ );
            SetCurrentThreadName( "ProgressBar" );

            instance.tryRunWithSehHandler_( [&instance, task]
            {
                instance.onFinish_ = task();
                return true;
            } );
            finish();

            unregisterThreadRootTimeRecord( instance.rootTimeRecord_ );
        } );
#else
        staticTaskForLaterCall = [&instance, task]
        {
            instance.tryRunWithSehHandler_( [&instance, task]
            {
                instance.onFinish_ = task();
                return true;
            } );
            finish();
        };
        emscripten_async_call( asyncCallTask, nullptr, 200 );
#endif
    };

    instance.initialize_( name, taskCount, postInit );

    getViewerInstance().incrementForceRedrawFrames();
}

void orderWithManualFinish( const char* name, std::function<void ()> task, int taskCount )
{
    auto& instance = ProgressBarImpl::instance();

    if ( isFinished() && instance.thread_.joinable() )
        instance.thread_.join();

    instance.isOrdered_ = true;

    auto postInit = [&instance, task]
    {
        // finalizer is not required
        instance.onFinish_ = {};
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
        instance.thread_ = std::thread( [&instance, task]
        {
            registerThreadRootTimeRecord( instance.rootTimeRecord_ );
            SetCurrentThreadName( "ProgressBar" );

            instance.tryRunWithSehHandler_( [task]
            {
                task();
                return true;
            } );

            unregisterThreadRootTimeRecord( instance.rootTimeRecord_ );
        } );
#else
        staticTaskForLaterCall = [&instance, task]
        {
            instance.tryRunWithSehHandler_( [task]
            {
                task();
                return true;
            } );
        };
        emscripten_async_call( asyncCallTask, nullptr, 200 );
#endif
    };

    instance.initialize_( name, taskCount, postInit );

    getViewerInstance().incrementForceRedrawFrames();
}

bool isCanceled()
{
    return ProgressBarImpl::instance().canceled_;
}

bool isFinished()
{
    return ProgressBarImpl::instance().finished_;
}

float getProgress()
{
    return ProgressBarImpl::instance().progress_;
}

float getLastOperationTime()
{
    return ProgressBarImpl::instance().lastOperationTimeSec_;
}

const std::string& getLastOperationTitle()
{
    return ProgressBarImpl::instance().title_;
}

bool setProgress( float p )
{
    auto& instance = ProgressBarImpl::instance();
    if ( p == instance.progress_ )
        return !instance.canceled_; // fast path if the progress value has not changed

    int newPercents = int( p * 100.0f );
    int percents = instance.percents_;
    if ( percents != newPercents && instance.percents_.compare_exchange_strong( percents, newPercents ) )
    {
        std::unique_lock lock( instance.mutex_ );
        spdlog::info( "Operation progress: \"{}\" - {}%", instance.title_, newPercents );
    }

    assert( p >= instance.progress_ ); // the progress must not jump backward
    assert( p <= 1 ); // the progress must not exceed 100%
    instance.progress_ = p;
    instance.frameRequest_.requestFrame();
    return !instance.canceled_;
}

void setTaskCount( int n )
{
    ProgressBarImpl::instance().taskCount_ = n;
}

void finish()
{
    auto& instance = ProgressBarImpl::instance();
    instance.finished_ = true;
    instance.frameRequest_.requestFrame();
}

bool isOrdered()
{
    return ProgressBarImpl::instance().isOrdered_;
}

void nextTask()
{
    auto& instance = ProgressBarImpl::instance();
    if ( instance.currentTask_ != instance.taskCount_ )
    {
        ++instance.currentTask_;
        callBackSetProgress( 0.0f );
    }
}

void nextTask( const char* s )
{
    {
        std::unique_lock lock( ProgressBarImpl::instance().mutex_ );
        ProgressBarImpl::instance().taskName_ = s;
    }
    nextTask();
}

void forceSetTaskName( std::string taskName )
{
    auto& instance = ProgressBarImpl::instance();
    std::unique_lock lock( instance.mutex_ );
    instance.taskName_ = std::move( taskName );
    instance.overrideTaskName_ = true;
}

void resetTaskName()
{
    auto& instance = ProgressBarImpl::instance();
    std::unique_lock lock( instance.mutex_ );
    instance.overrideTaskName_ = false;
    instance.taskName_.clear();
}

bool callBackSetProgress( float p )
{
    auto& instance = ProgressBarImpl::instance();

    // set instance.allowCancel_ = true but write in memory only if it was false
    bool expected = false;
    instance.allowCancel_.compare_exchange_strong( expected, true, std::memory_order_relaxed );

    setProgress( ( p + float( instance.currentTask_ - 1 ) ) / instance.taskCount_ );
    return !instance.canceled_;
}

bool simpleCallBackSetProgress( float p )
{
    auto& instance = ProgressBarImpl::instance();

    // set instance.allowCancel_ = false but write in memory only if it was true
    bool expected = true;
    instance.allowCancel_.compare_exchange_strong( expected, false, std::memory_order_relaxed );

    setProgress( ( p + float( instance.currentTask_ - 1 ) ) / instance.taskCount_ );
    return true; // no cancel
}

void printTimingTree( double minTimeSec )
{
    auto& instance = ProgressBarImpl::instance();
    instance.rootTimeRecord_.minTimeSec = minTimeSec;
    instance.rootTimeRecord_.printTree();
}

} //namespace ProgressBar

} //namespace MR
