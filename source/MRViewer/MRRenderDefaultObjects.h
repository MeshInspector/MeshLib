#pragma once
#include "MRViewer/MRRenderNameObject.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

// Combines all the default UI `IRenderObjects`.
using RenderDefaultUiObject = RenderObjectCombinator<RenderNameObject>;

// Simple render object to clear dirty flags in render objects combinations
class RenderResetDirtyComponent : public virtual IRenderObject
{
public:
    RenderResetDirtyComponent( const VisualObject& object ) :
        objPtr_{ &object } {}
    // Returns true if something was rendered, or false if nothing to render.
    virtual bool render( const ModelRenderParams& )
    {
        if ( objPtr_ )
            objPtr_->resetDirty();
        return true;
    }
    virtual void renderPicker( const ModelBaseRenderParams&, unsigned )
    {
        if ( objPtr_ )
            objPtr_->resetDirty();
    }    
    virtual size_t heapBytes() const { return 0; }
    virtual size_t glBytes() const { return 0; }
private:
    const VisualObject* objPtr_{ nullptr };
};

}
