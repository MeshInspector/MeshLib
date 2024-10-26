#include "MRProgressBar.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "ImGuiHelpers.h"
#include "MRRibbonButtonDrawer.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRTimeRecord.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include "MRRibbonConstants.h"
#include "MRUIStyle.h"
#include "MRCommandLoop.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonMenu.h"
#include <thread>
#include <GLFW/glfw3.h>
#include "MRViewer/MRUITestEngine.h"
#include "imgui_internal.h"

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
void ProgressBar::setup( float scaling )
{
    auto& instance = instance_();

    if ( instance.deferredInit_ )
        instance.initialize_();

    constexpr size_t bufSize = 256;
    char buf[bufSize];

    {
        std::unique_lock lock( instance.mutex_ );
        snprintf( buf, bufSize, "%s###GlobalProgressBarPopup", instance.title_.c_str() );
    }
    instance.setupId_ = ImGui::GetID( buf );
    const Vector2f windowSize( 440.0f * scaling, 144.0f * scaling );
    auto& viewer = getViewerInstance();
    ImGui::SetNextWindowPos( 0.5f * ( Vector2f( viewer.framebufferSize ) - windowSize ), ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    if ( ImGui::BeginModalNoAnimation( buf, nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
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
                snprintf( buf, bufSize, "%s (%d/%d)\n", instance.taskName_.c_str(), (int)instance.currentTask_, (int)instance.taskCount_ );
                ImGui::Text( "%s", buf );
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
                pushNotification( { .header = fmt::format( "{:.1f} sec", time < 5.e-3f ? 0.f : time ),
                                    .text = instance.title_, .type = NotificationType::Time,.tags = NotificationTags::Report } );
            }
            if ( instance.onFinish_ )
            {
                instance.onFinish_();
                instance.onFinish_ = {};
            }
            instance.isOrdered_ = false;
            instance.closeDialogNextFrame_ = true;
            getViewerInstance().incrementForceRedrawFrames();
        }
        ImGui::EndPopup();
    }
    instance.isInit_ = true;
}

void ProgressBar::onFrameEnd()
{
    // this is needed to prevent unexpected closing on progress bar window in:
    // ImGui::NewFrame() / ImGui::UpdateMouseMovingWindowNewFrame() / ImGui::FocusWindow()
    // that can happen if progress bar is ordered on clicking to the window
    // (for example on finish editing some InputFloat, clicking on window makes ImGui think this window is moving
    //  and close progress bar modal before it starts, task of progress bar is going but post-processing is not)
    auto& inst = instance_();
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

void ProgressBar::order( const char* name, const std::function<void()>& task, int taskCount )
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

void ProgressBar::orderWithMainThreadPostProcessing( const char* name, TaskWithMainThreadPostProcessing task, int taskCount )
{
    auto& instance = instance_();
    if ( !instance.isInit_ )
    {
        auto res = task();
        res();
        return;
    }

    if ( isFinished() && instance.thread_.joinable() )
        instance.thread_.join();

    instance.isOrdered_ = true;

    auto postInit = [&instance, task]
    {
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
        instance.thread_ = std::thread( [&instance, task]
        {
            static ThreadRootTimeRecord rootRecord( "Progress" );
            registerThreadRootTimeRecord( rootRecord );
            SetCurrentThreadName( "ProgressBar" );

            instance.tryRunWithSehHandler_( [&instance, task]
            {
                instance.onFinish_ = task();
                return true;
            } );
            ProgressBar::finish();

            unregisterThreadRootTimeRecord( rootRecord );
        } );
#else
        staticTaskForLaterCall = [&instance, task]
        {
            instance.tryRunWithSehHandler_( [&instance, task]
            {
                instance.onFinish_ = task();
                return true;
            } );
            ProgressBar::finish();
        };
        emscripten_async_call( asyncCallTask, nullptr, 200 );
#endif
    };

    instance.deferredInit_ = std::make_unique<DeferredInit>( DeferredInit {
        .taskCount = taskCount,
        .name = name,
        .postInit = postInit,
    } );

    getViewerInstance().incrementForceRedrawFrames();
}

void ProgressBar::orderWithManualFinish( const char* name, std::function<void ()> task, int taskCount )
{
    auto& instance = instance_();

    if ( !instance.isInit_ )
        return;

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
            static ThreadRootTimeRecord rootRecord( "Progress" );
            registerThreadRootTimeRecord( rootRecord );
            SetCurrentThreadName( "ProgressBar" );

            instance.tryRunWithSehHandler_( [task]
            {
                task();
                return true;
            } );

            unregisterThreadRootTimeRecord( rootRecord );
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

    instance.deferredInit_ = std::make_unique<DeferredInit>( DeferredInit {
        .taskCount = taskCount,
        .name = name,
        .postInit = postInit,
    } );

    getViewerInstance().incrementForceRedrawFrames();
}

bool ProgressBar::isCanceled()
{
    return instance_().canceled_;
}

bool ProgressBar::isFinished()
{
    return instance_().finished_;
}

float ProgressBar::getProgress()
{
    return instance_().progress_;
}

float ProgressBar::getLastOperationTime()
{
    return instance_().lastOperationTimeSec_;
}

const std::string& ProgressBar::getLastOperationTitle()
{
    return instance_().title_;
}

bool ProgressBar::setProgress( float p )
{
    auto& instance = instance_();

    // this assert is not really needed
    // leave it in comment: we don't expect progress from different threads
    //assert( instance.thread_.get_id() == std::this_thread::get_id() );

    int newPercents = int( p * 100.0f );
    int percents = instance.percents_;
    if ( percents != newPercents && instance.percents_.compare_exchange_strong( percents, newPercents ) )
    {
        std::unique_lock lock( instance.mutex_ );
        spdlog::info( "Operation progress: \"{}\" - {}%", instance.title_, newPercents );
    }

    instance.progress_ = p;
    instance.frameRequest_.requestFrame();
    return !instance.canceled_;
}

void ProgressBar::setTaskCount( int n )
{
    instance_().taskCount_ = n;
}

void ProgressBar::finish()
{
    auto& instance = instance_();
    instance.finished_ = true;
    instance.frameRequest_.requestFrame();
}

bool ProgressBar::isOrdered()
{
    return instance_().isOrdered_;
}

void ProgressBar::nextTask()
{
    auto& instance = instance_();
    if ( instance.currentTask_ != instance.taskCount_ )
    {
        ++instance.currentTask_;
        callBackSetProgress( 0.0f );
    }
}

void ProgressBar::nextTask( const char* s )
{
    {
        std::unique_lock lock( instance_().mutex_ );
        instance_().taskName_ = s;
    }
    nextTask();
}

void ProgressBar::forceSetTaskName( std::string taskName )
{
    auto& instance = instance_();
    std::unique_lock lock( instance.mutex_ );
    instance.taskName_ = std::move( taskName );
    instance.overrideTaskName_ = true;
}

void ProgressBar::resetTaskName()
{
    auto& instance = instance_();
    std::unique_lock lock( instance.mutex_ );
    instance.overrideTaskName_ = false;
    instance.taskName_.clear();
}

bool ProgressBar::callBackSetProgress( float p )
{
    auto& instance = instance_();
    instance.allowCancel_ = true;
    instance.setProgress( ( p + float( instance.currentTask_ - 1 ) ) / instance.taskCount_ );
    return !instance.canceled_;
}

bool ProgressBar::simpleCallBackSetProgress( float p )
{
    auto& instance = instance_();
    instance.allowCancel_ = false;
    instance.setProgress( ( p + float( instance.currentTask_ - 1 ) ) / instance.taskCount_ );
    return true; // no cancel
}

ProgressBar& ProgressBar::ProgressBar::instance_()
{
    static ProgressBar instance_;
    return instance_;
}

ProgressBar::ProgressBar() :
    progress_( 0.0f ), currentTask_( 0 ), taskCount_( 1 ),
    taskName_( "Current task" ), title_( "Sample Title" ),
    canceled_{ false }, finished_{ false }
{}

ProgressBar::~ProgressBar()
{
    canceled_ = true;
    if ( thread_.joinable() )
        thread_.join();
}

void ProgressBar::initialize_()
{
    assert( deferredInit_ );

    if ( isFinished() && thread_.joinable() )
        thread_.join();

    ImGui::CloseCurrentPopup();

    progress_ = 0.0f;

    taskCount_ = deferredInit_->taskCount;
    currentTask_ = 0;
    if ( taskCount_ == 1 )
        currentTask_ = 1;

    closeDialogNextFrame_ = false;
    canceled_ = false;
    finished_ = false;

    {
        std::unique_lock lock( mutex_ );
        title_ = deferredInit_->name;
    }

    ImGui::OpenPopup( setupId_ );
    frameRequest_.reset();

    operationStartTime_ = std::chrono::system_clock::now();
    if ( deferredInit_->postInit )
        deferredInit_->postInit();

    deferredInit_.reset();
}

bool ProgressBar::tryRun_( const std::function<bool ()>& task )
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
    catch ( const std::exception& e )
    {
        onFinish_ = [msg = std::string( e.what() )]
        {
            showError( msg );
        };
        return true;
    }
#endif
}

bool ProgressBar::tryRunWithSehHandler_( const std::function<bool()>& task )
{
#ifndef _WIN32
    return task();
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

}
