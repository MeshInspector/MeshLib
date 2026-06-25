#import "MRMacOSOpenDocumentsHandler.h"
#import <objc/runtime.h>

// callback function implemented in .cpp
// accumulates file names until called with null, then handles the list
extern "C" void handle_load_message(const char* filePath);

// based on
// https://github.com/glfw/glfw/issues/1024#issuecomment-522667555
@implementation GLFWCustomDelegate

+ (void)load{
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        Class class_  = objc_getClass("GLFWApplicationDelegate");
    
        [GLFWCustomDelegate swizzle:class_ src:@selector(application:openFile:) tgt:@selector(swz_application:openFile:)];
        [GLFWCustomDelegate swizzle:class_ src:@selector(application:openFiles:) tgt:@selector(swz_application:openFiles:)];
        // This call makes it so that application:openFile: doesn't get bogus calls
        // from Cocoa doing its own parsing of the argument string. And yes, we need
        // to use a string with a boolean value in it. That's just how it works.
        [[NSUserDefaults standardUserDefaults]
            setObject:@"NO"
               forKey:@"NSTreatUnknownArgumentsAsOpen"];
    });
}

+ (void) swizzle:(Class) original_c src:(SEL)original_s tgt:(SEL)target_s{
    Class target_c = [GLFWCustomDelegate class];
    Method originalMethod = class_getInstanceMethod(original_c, original_s);
    Method swizzledMethod = class_getInstanceMethod(target_c, target_s);

    BOOL didAddMethod =
    class_addMethod(original_c,
                    original_s,
                    method_getImplementation(swizzledMethod),
                    method_getTypeEncoding(swizzledMethod));

    if (didAddMethod) {
        class_replaceMethod(original_c,
                            target_s,
                            method_getImplementation(originalMethod),
                            method_getTypeEncoding(originalMethod));
    } else {
        method_exchangeImplementations(originalMethod, swizzledMethod);
    }
}

- (BOOL)swz_application:(NSApplication *)sender openFile:(NSString *)filename{
    handle_load_message(filename.UTF8String);
    handle_load_message(NULL);
    return TRUE;
}

- (void)swz_application:(NSApplication *)sender openFiles:(NSArray<NSString *> *)filenames{
    for (NSString *nsString in filenames)
        handle_load_message(nsString.UTF8String);
    handle_load_message(NULL);
}

@end
