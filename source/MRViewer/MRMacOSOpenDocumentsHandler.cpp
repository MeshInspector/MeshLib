#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer.h"
#include "MRProgressBar.h"
#include "MRViewerInstance.h"
#include "MRRibbonMenu.h"
#include "MRCommandLoop.h"

#ifdef __APPLE__

// callback function from .mm
// accumulates file names until called with null, then handles the list
extern "C" void handle_load_message( const char* filePath )
{
    static std::vector<std::filesystem::path> filePaths;
    static std::string joined;
    if ( filePath )
    {
        filePaths.push_back( MR::pathFromUtf8( filePath ) );
        if ( !joined.empty() )
            joined += ',';
        joined += filePath;
    }
    else
    {
        spdlog::info( "Open file(s) requested: {}", joined );
        MR::CommandLoop::appendCommand( [paths = filePaths] ()
        {
            auto& viewerRef = MR::getViewerInstance();
            auto menu = viewerRef.getMenuPluginAs<MR::RibbonMenu>();
            if ( MR::ProgressBar::isOrdered() )
            {
                if ( menu )
                    menu->pushNotification( { .text = "Another operation in progress.", .lifeTimeSec = 3.0f } );
                return;
            }
            viewerRef.loadFiles( paths );
        }, MR::CommandLoop::StartPosition::AfterPluginInit );
        joined.clear();
        filePaths.clear();
    }
}

#endif
