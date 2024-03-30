#pragma once
#include "MRFrameRedrawRequest.h"
#include <imgui.h>
#include <functional>
#include <atomic>
#include <thread>

namespace MR
{

// This class shows application progress bar for long operations
// note! if class don't setup, then order and orderWithMainThreadPostProcessing methods call task directly
class ProgressBar
{
public:
    using TaskWithMainThreadPostProcessing = std::function< std::function<void()>() >;
    // this function should be called only once for each frame (it is called in MR::Menu (MR::RibbonMenu))
    MRVIEWER_API static void setup( float scaling );
    // call this function on frame end
    MRVIEWER_API static void onFrameEnd();

    // this shall be called in order to start concurrent task execution with progress bar display
    MRVIEWER_API static void order(const char * name, const std::function<void()>& task, int taskCount = 1 );

    // in this version the task returns a function to be executed in main thread
    MRVIEWER_API static void orderWithMainThreadPostProcessing( const char* name, TaskWithMainThreadPostProcessing task, int taskCount = 1 );

    /// the task is spawned by the progress bar but the `finish` method is called from a callback
    MRVIEWER_API static void orderWithManualFinish( const char * name, std::function<void ()> task, int taskCount = 1 );

    MRVIEWER_API static bool isCanceled();

    MRVIEWER_API static bool isFinished();

    MRVIEWER_API static float getProgress();

    // returns time of last operation in seconds, returns -1.0f if no operation was performed
    MRVIEWER_API static float getLastOperationTime();
    // returns title of the last operation
    MRVIEWER_API static const std::string& getLastOperationTitle();

    // sets the current progress and returns false if the user has pressed Cancel button
    MRVIEWER_API static bool setProgress(float p);

    MRVIEWER_API static void nextTask();
    MRVIEWER_API static void nextTask(const char * s);

    MRVIEWER_API static void setTaskCount( int n );

    // set the current task's name without auto-updating progress value
    MRVIEWER_API static void forceSetTaskName( std::string taskName );
    MRVIEWER_API static void resetTaskName();

    MRVIEWER_API static void finish();

    // returns true if progress bar was ordered and not finished
    MRVIEWER_API static bool isOrdered();

    // these callbacks allow canceling
    MRVIEWER_API static bool callBackSetProgress(float p);
    // these callbacks do not allow canceling
    MRVIEWER_API static bool simpleCallBackSetProgress( float p );
private:
    static ProgressBar& instance_();

    ProgressBar();
    ~ProgressBar();

    void initialize_();

    // cover task execution with try catch block (in release only)
    // if catches exception shows error in main thread overriding user defined main thread post-processing
    bool tryRun_( const std::function<bool ()>& task );
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

    // needed to be able to call progress bar from any point, not only from ImGui frame scope
    struct DeferredInit
    {
        int taskCount;
        std::string name;
        std::function<void ()> postInit;
    };
    std::unique_ptr<DeferredInit> deferredInit_;

    std::atomic<bool> allowCancel_;
    std::atomic<bool> canceled_;
    std::atomic<bool> finished_;
    ImGuiID setupId_ = ImGuiID( -1 );

    bool isOrdered_{ false };
    bool isInit_{ false };
    // this is needed to show full progress before closing
    bool closeDialogNextFrame_{ false };
};

}
