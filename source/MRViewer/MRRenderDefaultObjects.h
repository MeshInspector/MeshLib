#pragma once
#include "MRViewer/MRRenderNameObject.h"

namespace MR
{

class VisualObject;

// Combines all the default UI `IRenderObjects`.
using RenderDefaultUiObject = RenderObjectCombinator<RenderNameObject>;

// Simple render object to clear dirty flags in render objects combinations
class RenderResetDirtyComponent : public virtual IRenderObject
{
public:
    MRVIEWER_API RenderResetDirtyComponent( const VisualObject& object );
    // only clears dirty flag of the object
    MRVIEWER_API virtual bool render( const ModelRenderParams& );
    MRVIEWER_API virtual void renderPicker( const ModelBaseRenderParams&, unsigned );

    virtual size_t heapBytes() const { return 0; }
    virtual size_t glBytes() const { return 0; }
private:
    const VisualObject* objPtr_{ nullptr };
};

}
