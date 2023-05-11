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
#include <GLFW/glfw3.h>

#ifdef _WIN32
#include <excpt.h>
#endif

#if defined( __EMSCRIPTEN__ ) && !defined( __EMSCRIPTEN_PTHREADS__ )
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

namespace MR
{
void ProgressBar::setup( float scaling )
{
    auto& instance = instance_();

    if ( instance.deferredProgressBar_ )
    {
        instance.deferredProgressBar_();
        instance.deferredProgressBar_ = {};
    }

    constexpr size_t bufSize = 256;
    char buf[bufSize];

    snprintf( buf, bufSize, "%s###GlobalProgressBarPopup", instance.title_.c_str() );
    instance.setupId_ = ImGui::GetID( buf );
    if ( ImGui::BeginModalNoAnimation( buf, nullptr, ImGuiWindowFlags_AlwaysAutoResize ) )
    {
        instance.frameRequest_.reset();

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
        if ( instance.taskCount_ > 1 )
        {
            snprintf( buf, bufSize, "%s (%d/%d)\n", instance.taskName_.c_str(), instance.currentTask_, instance.taskCount_ );
            ImGui::Text( "%s", buf );
        }

        snprintf( buf, bufSize, "%d%%", ( int )( instance.progress_ * 100 ) );
        auto progress = instance.progress_;
        ImGui::ProgressBar( progress, ImVec2( 250.0f * scaling, 0.0f ), buf );
        // this is needed to prevent events race and redraw after progress bar is finished
        if ( progress >= 1.0f )
            instance.frameRequest_.requestFrame();
        ImGui::Separator();

        if ( instance.allowCancel_ )
        {
            if ( !instance.canceled_ )
            {
                ImGui::SetCursorPosX( ( ImGui::GetWindowWidth() + ImGui::GetContentRegionAvail().x ) * 0.5f - 75.0f * scaling );
				const float btnHeight = ImGui::CalcTextSize( "SDC" ).y + cGradientButtonFramePadding * scaling;
                if ( UI::button( "Cancel", Vector2f( 75.0f * scaling, btnHeight ), ImGuiKey_Escape ) )
                    instance.canceled_ = true;
            }
            else
            {
                ImGui::AlignTextToFramePadding();
                ImGui::Text( "Canceling..." );
            }
        }
#else
        ImGui::Text( "Operation is in progress, please wait..." );
        if ( instance.progress_ >= 1.0f )
            instance.frameRequest_.requestFrame();
#endif
        if ( instance.finished_ )
        {
            if ( instance.onFinish_ )
            {
                instance.onFinish_();
                instance.onFinish_ = {};
            }
            ImGui::CloseCurrentPopup();
            getViewerInstance().incrementForceRedrawFrames( 2, true );
        }
        ImGui::EndPopup();
    }
    instance.isInit_ = true;
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
    {
        instance.thread_.join();
    }

    if ( instance.thread_.joinable() )
        return;

    instance.deferredProgressBar_ = [task, taskCount, nameStr = std::string(name)] ()
    {
        auto& instance = instance_();

        if ( isFinished() && instance.thread_.joinable() )
            instance.thread_.join();

        if ( instance.thread_.joinable() )
            return;

        instance.progress_ = 0.0f;

        instance.task_ = task;
        instance.currentTask_ = 0;
        if ( taskCount == 1 )
            instance.currentTask_ = 1;
        instance.taskCount_ = taskCount;

        instance.canceled_ = false;
        instance.finished_ = false;

        instance.title_ = nameStr;

        ImGui::OpenPopup( instance.setupId_ );
        instance.frameRequest_.reset();
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
        instance.thread_ = std::thread( [&instance] ()
        {
            static ThreadRootTimeRecord rootRecord( "Progress" );
            registerThreadRootTimeRecord( rootRecord );
            SetCurrentThreadName( "ProgressBar" );
            instance.tryRunTaskWithSehHandler_();
            unregisterThreadRootTimeRecord( rootRecord );
        } );
#else
        staticTaskForLaterCall = [&instance] () 
        {
            instance.tryRunTaskWithSehHandler_();
        };
        emscripten_async_call( asyncCallTask, nullptr, 200 );
#endif
    };
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

bool ProgressBar::setProgress( float p )
{
    auto& instance = instance_();
    assert( instance.thread_.get_id() == std::this_thread::get_id() );
    int newPercents = int( p * 100.0f );
    int percents = instance.percents_;
    if ( percents != newPercents && instance.percents_.compare_exchange_strong( percents, newPercents ) )
        spdlog::info( "Operation progress: \"{}\" - {}%", instance.title_, newPercents );

    instance.progress_ = p;
    instance.frameRequest_.requestFrame();
    return !instance.canceled_;
}

void ProgressBar::setTaskCount( int n )
{
    instance_().taskCount_ = n;
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
    instance_().taskName_ = s;
    nextTask();
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

void ProgressBar::tryRunTask_()
{
#ifndef NDEBUG
    onFinish_ = task_();
#else
    try
    {
        onFinish_ = task_();
    }
    catch ( const std::bad_alloc& badAllocE )
    {
        onFinish_ = [msg = std::string( badAllocE.what() )]()
        {
            spdlog::error( msg );
            if ( auto menu = getViewerInstance().getMenuPlugin() )
                menu->showErrorModal( "Device ran out of memory during this operation." );
        };
    }
    catch ( const std::exception& e )
    {
        onFinish_ = [msg = std::string( e.what() )]()
        {
            spdlog::error( msg );
            if ( auto menu = getViewerInstance().getMenuPlugin() )
                menu->showErrorModal( msg );
        };
    }
#endif
}

void ProgressBar::tryRunTaskWithSehHandler_()
{
#if !defined(_WIN32) || !defined(NDEBUG)
    tryRunTask_();
#else
    __try
    {
        tryRunTask_();
    }
    __except ( EXCEPTION_EXECUTE_HANDLER )
    {
        onFinish_ = []()
        {
            spdlog::error( "Unknown exception occurred" );
            if ( auto menu = getViewerInstance().getMenuPlugin() )
                menu->showErrorModal( "Unknown exception occurred" );
        };
    }
#endif
    finish_();
}

void ProgressBar::FrameRedrawRequest::requestFrame()
{
    // do not do it too frequently not to overload the renderer
    constexpr auto minInterval = std::chrono::milliseconds( 100 );
    // make request
#ifdef __EMSCRIPTEN__
    bool testFrameOrder = false;
    if ( frameRequested_.compare_exchange_strong( testFrameOrder, true ) )
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
        MAIN_THREAD_EM_ASM( postEmptyEvent( $0, 2 ), int( minInterval.count() ) );
#pragma clang diagnostic pop
    }
#else
    asyncRequest_.requestIfNotSet( std::chrono::system_clock::now() + minInterval, [] ()
    {
        getViewerInstance().postEmptyEvent();
    } );
#endif
}

void ProgressBar::finish_()
{
    finished_ = true;
    frameRequest_.requestFrame();
}

void ProgressBar::FrameRedrawRequest::reset()
{
#ifdef __EMSCRIPTEN__
    bool testFrameOrder = true;
    frameRequested_.compare_exchange_strong( testFrameOrder, false );
#endif
}

}
