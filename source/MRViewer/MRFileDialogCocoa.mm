#include "MRFileDialogCocoa.h"

#include "MRMesh/MRFinally.h"

#include <AppKit/AppKit.h>

namespace
{

inline std::string fromNSURL( NSURL* url )
{
    return [[url path] UTF8String];
}

inline NSString* toNSString( const std::string& path )
{
    return [NSString stringWithUTF8String:path.c_str()];
}

inline NSURL* toNSURL( const std::string& path )
{
    return [NSURL fileURLWithPath:toNSString( path )];
}

} // namespace

namespace MR::detail
{

std::vector<std::filesystem::path> runCocoaFileDialog( const FileDialogParameters& params )
{
    // enable garbage collector
    auto* pool = [[NSAutoreleasePool alloc] init];
    MR_FINALLY {
        [pool release];
    };

    // set focus back to the main window at the end
    auto* keyWindow = [[NSApplication sharedApplication] keyWindow];
    MR_FINALLY {
        [keyWindow makeKeyAndOrderFront:nil];
    };

    NSSavePanel* dialog;
    if ( params.saveDialog )
    {
        dialog = [NSSavePanel savePanel];
    }
    else
    {
        auto* openDialog = [NSOpenPanel openPanel];

        [openDialog setAllowsMultipleSelection:params.multiselect];
        [openDialog setCanChooseDirectories:params.folderDialog];
        [openDialog setCanChooseFiles:!params.folderDialog];

        dialog = openDialog;
    }

    const auto currentFolder = getCurrentFolder( params.baseFolder );
    if ( !currentFolder.empty() )
        [dialog setDirectoryURL:toNSURL( currentFolder )];

    if ( !params.fileName.empty() )
        [dialog setNameFieldStringValue:toNSString( params.fileName )];

    if ( [dialog runModal] != NSModalResponseOK )
        return {};

    std::vector<std::filesystem::path> results;
    if ( !params.saveDialog && params.multiselect )
    {
        auto* urls = [(NSOpenPanel*)dialog URLs];
        for ( NSURL* url in urls )
            results.emplace_back( fromNSURL( url ) );
    }
    else
    {
        auto* url = [dialog URL];
        results.emplace_back( fromNSURL( url ) );
    }
    return results;
}

} // namespace MR::detail
