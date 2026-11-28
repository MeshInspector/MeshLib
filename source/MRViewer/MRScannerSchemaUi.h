#pragma once

#include "exports.h"
#include "MRScannerBackend.h"

#include <functional>
#include <memory>

namespace MR
{

/// The generic, backend-agnostic scanner UI: renders a backend's parameter schema (paramSchema)
/// as ImGui widgets and dispatches its actions through the interface. Each part is
/// drawn separately so the host can lay them out (e.g. in tabs); all must be called inside an
/// already-open window. `onScan` is invoked with any VisualObject a capture or an action yields,
/// so the host can insert/post-process it. `error` is the host's message of the last failure: each
/// operation started here sets it when it fails and clears it when it succeeds, as does Disconnect.
using ScannerOnScan = std::function<void( std::shared_ptr<VisualObject> )>;

/// The schema's parameter groups.
MRVIEWER_API void drawScannerParams( IScannerBackend& backend );

/// The schema's backend-specific action groups, each with its status line.
/// Returns true when an action ran, e.g. to refresh a preview of the backend's state.
MRVIEWER_API bool drawScannerActions( IScannerBackend& backend, const ScannerOnScan& onScan, std::string& error );

/// The common verbs: capture (also on Space), connect / disconnect and live preview as the
/// backend's capabilities allow, then the error.
/// Returns true when a capture ran, successful or not, e.g. to refresh a preview of the backend's state.
MRVIEWER_API bool drawScannerCapture( IScannerBackend& backend, const ScannerOnScan& onScan, std::string& error );

/// A block title, with the tighter spacing the scanner tools use around it.
MRVIEWER_API void drawScannerSeparator( const std::string& title );

} // namespace MR
