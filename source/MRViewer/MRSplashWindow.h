#pragma once

#include "MRViewerFwd.h"
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <memory>

struct GLFWwindow;
struct ImGuiContext;

namespace MR
{

// Base class to run splash window in new thread
// override it's private functions and provide it Viever::LaunchParams
class MRVIEWER_CLASS SplashWindow
{
public:
    MRVIEWER_API SplashWindow( std::string name );

    // Thread should be stopped before destructor
    MRVIEWER_API virtual ~SplashWindow();

    // Starts splash window in new thread, non blocking
    // window will be closed in destructor of this class, or if `frame_` func returns false. or manually with `stop` function
    MRVIEWER_API void start();

    // Closes splash window if it is still opened
    MRVIEWER_API void stop();

    // Returns minimum time in seconds, splash screen to be present
    virtual float minimumTimeSec() const { return 5.0f; }

protected:
    std::string name_;

    GLFWwindow* window_{ nullptr };
    ImGuiContext* guiContext_{ nullptr };

    std::atomic<bool> terminate_{ false };
private:
    // This function is called in main thread right after splash thread is started
    virtual void afterStart_() {}
    // This function is called in main thread right before splash thread is stopped
    virtual void beforeStop_() {}

    // This is called first (for window decoration, etc.), splash thread
    virtual void setup_() const = 0;
    // This is called right after window is created to make some initial actions (for example load splash texture), splash thread
    virtual void postInit_() {}
    // This is called after 'postInit_' (and on dpi change) to resize and reposition window, splash thread
    virtual void positioning_( float hdpiScale ) = 0;
    // This is called after 'positioning_' to reload font, splash thread
    virtual void reloadFont_( float hdpiScale, float pixelRatio ) = 0;
    // This is called each frame, return false to close splash, splash thread
    virtual bool frame_( float scaling ) = 0;
    // This is called right before GLWF and ImGui contexts are destructed (needed to clear some GL data), splash thread
    virtual void preDestruct_() {}
    // This is called right after GLWF and ImGui contexts are destructed and right before closing splash (restore Menu ImGui context, etc.), splash thread
    virtual void postDestruct_() {}

    std::thread thread_;
};

#ifndef __EMSCRIPTEN__
class MRVIEWER_CLASS DefaultSplashWindow final : public SplashWindow
{
public:
    MRVIEWER_API DefaultSplashWindow();
private:
    virtual void setup_() const override;
    virtual void postInit_() override;
    virtual void positioning_( float hdpiScale ) override;
    virtual void reloadFont_( float hdpiScale, float pixelRatio ) override;
    virtual bool frame_( float scaling ) override;
    virtual void preDestruct_() override;

    std::shared_ptr<ImGuiImage> splashImage_;
    std::string versionStr_;
};
#endif
}