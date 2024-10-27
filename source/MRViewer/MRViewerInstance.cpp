#include "MRViewerInstance.h"
#include "MRViewer.h"

namespace MR
{

Viewer& getViewerInstance()
{
    static Viewer viewer;
    return viewer;
}

void incrementForceRedrawFrames( int i, bool swapOnLastOnly )
{
    getViewerInstance().incrementForceRedrawFrames( i, swapOnLastOnly );
}

} //namespace MR
