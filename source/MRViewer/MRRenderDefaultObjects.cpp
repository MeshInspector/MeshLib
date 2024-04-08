#include "MRRenderDefaultObjects.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

RenderResetDirtyComponent::RenderResetDirtyComponent( const VisualObject& object ) :
    objPtr_{ &object }
{}

bool RenderResetDirtyComponent::render( const ModelRenderParams& )
{
    if ( objPtr_ )
        objPtr_->resetDirty();
    return true;
}

void RenderResetDirtyComponent::renderPicker( const ModelBaseRenderParams&, unsigned )
{
    if ( objPtr_ )
        objPtr_->resetDirty();
}

}