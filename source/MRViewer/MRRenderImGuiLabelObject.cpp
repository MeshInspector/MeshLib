#include "MRRenderImGuiLabelObject.h"
#include "MRObjectImGuiLabel.h"

#include "MRPch/MRFmt.h"

namespace MR
{

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectImGuiLabel, RenderImGuiLabelObject )

RenderImGuiLabelObject::RenderImGuiLabelObject( const VisualObject& object )
    : RenderObjectCombinator( object ),
    object_( &dynamic_cast<const ObjectImGuiLabel&>( object ) )
{}

void RenderImGuiLabelObject::renderUi( const UiRenderParams& params )
{
    Vector3f point = object_->xf().b;
    if ( auto p = object_->parent() )
        point = p->worldXf()( point );

    task_ = UI::NonOverlappingLabelTask( {
        .menuScaling = params.scale,
        .viewportId = params.viewportId,
        .worldPos = point,
        .id = fmt::format( "{}", (void*)this ),
        .text = object_->getLabel(),
    } );

    params.tasks->push_back( { std::shared_ptr<void>{}, &task_ } );
}

}
