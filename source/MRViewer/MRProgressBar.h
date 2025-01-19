#pragma once

#include "exports.h"
#include <functional>
#include <string>

namespace MR
{

// Utilities to show application progress bar for long operations
namespace ProgressBar
{

/// function that returns post-processing function to be called in main UI thread
using TaskWithMainThreadPostProcessing = std::function< std::function<void()>() >;

/// this function should be called only once for each frame (it is called in MR::Menu (MR::RibbonMenu))
MRVIEWER_API  void setup( float scaling );

/// call this function on frame end
MRVIEWER_API  void onFrameEnd();

/// this shall be called in order to start concurrent task execution with progress bar display
/// please call setup() first, otherwise this function just execute task directly
MRVIEWER_API  void order(const char * name, const std::function<void()>& task, int taskCount = 1 );

/// in this version the task returns a function to be executed in main thread
/// please call setup() first, otherwise this function just execute task directly
MRVIEWER_API  void orderWithMainThreadPostProcessing( const char* name, TaskWithMainThreadPostProcessing task, int taskCount = 1 );

/// the task is spawned by the progress bar but the `finish` method is called from a callback
/// please call setup() first, otherwise this function just execute task directly
MRVIEWER_API  void orderWithManualFinish( const char * name, std::function<void ()> task, int taskCount = 1 );

MRVIEWER_API  bool isCanceled();

MRVIEWER_API  bool isFinished();

MRVIEWER_API  float getProgress();

/// returns time of last operation in seconds, returns -1.0f if no operation was performed
MRVIEWER_API  float getLastOperationTime();

/// returns title of the last operation
MRVIEWER_API  const std::string& getLastOperationTitle();

/// sets the current progress and returns false if the user has pressed Cancel button
MRVIEWER_API  bool setProgress(float p);

MRVIEWER_API  void nextTask();
MRVIEWER_API  void nextTask(const char * s);

MRVIEWER_API  void setTaskCount( int n );

/// set the current task's name without auto-updating progress value
MRVIEWER_API  void forceSetTaskName( std::string taskName );
MRVIEWER_API  void resetTaskName();

MRVIEWER_API  void finish();

/// returns true if progress bar was ordered and not finished
MRVIEWER_API  bool isOrdered();

/// these callbacks allow canceling
MRVIEWER_API  bool callBackSetProgress(float p);

/// these callbacks do not allow canceling
MRVIEWER_API  bool simpleCallBackSetProgress( float p );

/// prints time tree of progress bar thread
/// \param minTimeSec omit printing records with time spent less than given value in seconds
MRVIEWER_API  void printTimingTree( double minTimeSec = 0.1 );

} //namespace ProgressBar

} //namespace MR
