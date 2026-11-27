#include "MRViewer/MRScannerPlugin.h"
#include "MRViewer/MRRibbonRegisterItem.h"

namespace MR
{

// the Scanner tool over the backends registered by ScannerBackendAdder
static RibbonMenuItemAdder sScannerPlugin( makeScannerPlugin() );

} // namespace MR
