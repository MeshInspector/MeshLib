#pragma once

#include "MRViewerFwd.h"
#include "MRScannerBackend.h"

namespace MR
{

/// Creates the Scanner tool: it lists the given backends, communicates with the selected one only through
/// IScannerBackend, renders its parameter schema, and processes the scans (clip / rotate / register, scene insertion);
/// register it e.g. with RibbonMenuItemAdder from the module of the backends; returns null without backends
MRVIEWER_API std::shared_ptr<RibbonMenuItem> makeScannerPlugin( std::vector<std::unique_ptr<IScannerBackend>> backends );

} // namespace MR
