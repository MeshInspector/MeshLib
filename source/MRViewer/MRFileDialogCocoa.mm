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

    if ( !params.folderDialog && !params.filters.empty() )
    {
        auto* fileTypes = [[NSMutableArray alloc] init];
        for ( const auto& filter : params.filters )
        {
            size_t separatorPos = 0;
            for (;;)
            {
                auto nextSeparatorPos = filter.extensions.find( ";", separatorPos );
                auto ext = filter.extensions.substr( separatorPos, nextSeparatorPos - separatorPos );

                assert( ext.starts_with( "*." ) );
                if ( ext != "*.*" )
                    [fileTypes addObject:toNSString( ext.substr( 2 ) )];

                if ( nextSeparatorPos == std::string::npos )
                    break;
                separatorPos = nextSeparatorPos + 1;
            }
        }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        [dialog setAllowedFileTypes:fileTypes];
#pragma clang diagnostic pop
    }

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
