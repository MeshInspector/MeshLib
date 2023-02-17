#pragma once
#include "exports.h"
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

    // this shall be called in order to start concurrent task execution with progress bar display
    MRVIEWER_API static void order(const char * name, const std::function<void()>& task, int taskCount = 1 );

    // in this version the task returns a function to be executed in main thread
    MRVIEWER_API static void orderWithMainThreadPostProcessing( const char* name, TaskWithMainThreadPostProcessing task, int taskCount = 1 );

    MRVIEWER_API static bool isCanceled();

    MRVIEWER_API static bool isFinished();

    MRVIEWER_API static bool isReady();

    MRVIEWER_API static float getProgress();

    // sets the current progress and returns false if the user has pressed Cancel button
    MRVIEWER_API static bool setProgress(float p);

    MRVIEWER_API static void addProgress(float p);

    MRVIEWER_API static void nextTask();
    MRVIEWER_API static void nextTask(const char * s);

    MRVIEWER_API static void setTaskCount( int n );

    // these callbacks allow canceling
    MRVIEWER_API static bool callBackSetProgress(float p);
    MRVIEWER_API static bool callBackAddProgress(float p);
    // these callbacks do not allow canceling
    MRVIEWER_API static bool simpleCallBackSetProgress( float p );
    MRVIEWER_API static bool simpleCallBackAddProgress( float p );
private:
    static ProgressBar& instance_();

    ProgressBar();
    ~ProgressBar();

    // cover task execution with try catch block (in release only)
    // if catches exception shows error in main thread overriding user defined main thread post processing
    void tryRunTask_();
    void tryRunTaskWithSehHandler_();

    void postEvent_();
    void finish_();

    float progress_;
    int currentTask_, taskCount_;
    std::string taskName_, title_;
    std::chrono::time_point<std::chrono::system_clock> lastPostEvent_;

    std::thread thread_;
    TaskWithMainThreadPostProcessing task_;
    std::function<void()> onFinish_;

    // needed to be able to call progress bar from any point, not only from ImGui frame scope
    std::function<void()> deferredProgressBar_;

    std::atomic<bool> allowCancel_;
    std::atomic<bool> canceled_;
    std::atomic<bool> finished_;
    ImGuiID setupId_ = ImGuiID( -1 );

    bool isReady_{ true };
    bool isInit_{ false };
};

}