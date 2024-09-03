#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

#ifdef __APPLE__

// A trick to set custom handler for openFile&openFiles in GLFW's class
// https://github.com/glfw/glfw/issues/1024#issuecomment-522667555
@interface GLFWCustomDelegate : NSObject
+ (void)load; // load is called before even main() is run (as part of objc class registration)
@end

#endif
