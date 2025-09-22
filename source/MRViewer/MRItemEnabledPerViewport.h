#pragma once

#include "MRMesh/MRViewportId.h"
#include "exports.h"

namespace MR
{

// A base class for `RibbonMenuItem`s that can be toggled per viewport.
// The intent is that the derived classes should also be derived from `RibbonMenuItem` and be registered in the ribbon schema.
// Doing this ensures that you can loop over all items in the schema to get a list of all items toggleable in this way.
class ItemEnabledPerViewport
{
public:
    [[nodiscard]] MRVIEWER_API ViewportMask getEnabledViewports() const;
    MRVIEWER_API void setEnabledViewports( ViewportMask newMask );

protected:
    ItemEnabledPerViewport() = default;

private:
    ViewportMask enabledViewports_ = ViewportMask::all();
};

}
