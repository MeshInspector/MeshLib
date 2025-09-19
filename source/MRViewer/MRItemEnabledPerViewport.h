#pragma once

#include "MRMesh/MRViewportId.h"
#include "MRViewer/MRRibbonMenuItem.h"

namespace MR
{

// A base class for `RibbonMenuItem`s that can be toggled per viewport.
class ItemEnabledPerViewport : public virtual RibbonMenuItem
{
public:
    [[nodiscard]] MRVIEWER_API ViewportMask getEnabledViewports() const;
    MRVIEWER_API void setEnabledViewports( ViewportMask newMask );

protected:
    MRVIEWER_API ItemEnabledPerViewport();

private:
    ViewportMask enabledViewports_ = ViewportMask::all();
};

}
