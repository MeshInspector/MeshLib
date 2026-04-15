#include "MRMcp/MRMcp.h"
#include "MRMesh/MROnInit.h"
#include "MRViewer/MRCommandLoop.h"

namespace MR
{

MR_ON_INIT{
    MR::CommandLoop::appendCommand( []
    {
        Mcp::getDefaultServer().setRunning( true );
    } );
};

} // namespace MR
