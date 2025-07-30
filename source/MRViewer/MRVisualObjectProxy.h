#pragma once

#include "MRViewerFwd.h"

#include "MRMesh/MRViewportId.h"

namespace MR
{

/// methods for overriding visual object properties based on its state (e.g. metadata values) or the app's state (e.g. current plugin)
class VisualObjectProxy
{
public:
    /// returns color of given object
    MRVIEWER_API static Color getFrontColor( const VisualObject& visObj, bool selected = true, ViewportId viewportId = {} );
};

} // namespace MR
