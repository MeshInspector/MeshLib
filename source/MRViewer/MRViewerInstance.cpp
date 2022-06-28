#include "MRViewerInstance.h"
#include "MRViewer.h"

namespace MR
{

Viewer& getViewerInstance()
{
    static Viewer viewer;
    return viewer;
}

} //namespace MR
