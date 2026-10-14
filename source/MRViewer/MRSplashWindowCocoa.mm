#include "MRSplashWindow.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRPch/MRSpdlog.h"

#include <AppKit/AppKit.h>
#include <QuartzCore/QuartzCore.h>

#include <algorithm>

namespace MR
{

void DefaultSplashWindow::start()
{
    @autoreleasepool
    {
        const auto path = SystemPath::getResourcesDirectory() / "MRSplash.png";
        NSImage* image = [[[NSImage alloc] initWithContentsOfFile:[NSString stringWithUTF8String:utf8string( path ).c_str()]] autorelease];
        NSImageRep* rep = image.representations.firstObject;
        if ( !rep || rep.pixelsWide <= 0 || rep.pixelsHigh <= 0 )
        {
            spdlog::error( "No splash image found" );
            return;
        }

        NSScreen* screen = [NSScreen mainScreen];
        if ( !screen )
            return;
        const NSRect workArea = screen.visibleFrame;
        const CGFloat width = std::min( 0.6 * workArea.size.width, CGFloat( rep.pixelsWide ) );
        const CGFloat height = width * rep.pixelsHigh / rep.pixelsWide;
        const NSRect frame = NSMakeRect(
            NSMidX( workArea ) - width / 2,
            NSMidY( workArea ) - height / 2,
            width, height );

        NSPanel* panel = [[NSPanel alloc] initWithContentRect:frame
                                                    styleMask:NSWindowStyleMaskBorderless
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];
        panel.releasedWhenClosed = NO;
        panel.opaque = NO;
        panel.backgroundColor = NSColor.clearColor;
        panel.hasShadow = NO;
        panel.level = NSFloatingWindowLevel;

        NSImageView* view = [NSImageView imageViewWithImage:image];
        view.imageScaling = NSImageScaleProportionallyUpOrDown;
        view.frame = NSMakeRect( 0, 0, width, height );
        panel.contentView = view;

        [panel orderFrontRegardless];
        [panel display];
        // the main thread is busy until stop(), so commit the drawing now instead of on the next run loop pass
        [CATransaction flush];

        nsWindow_ = panel;
        spdlog::info( "Splash window shown" );
    }
}

void DefaultSplashWindow::stop()
{
    if ( !nsWindow_ )
        return;
    NSPanel* panel = (NSPanel*)nsWindow_;
    nsWindow_ = nullptr;
    [panel orderOut:nil];
    [panel release];
    spdlog::info( "Splash window closed" );
}

} // namespace MR
