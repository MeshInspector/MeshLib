#pragma once

#include "MRViewer/MRRenderNameObject.h"

namespace MR
{

// This combines all the different UI `IRenderObject`s that most `VisualObject`s should have by default.
class RenderDefaultUiObject : public RenderNameObject
{
    using RenderNameObject::RenderNameObject;
};

// Combines template argument `Base` and `RenderDefaultUiObject`.
template <typename Base>
class RenderDefaultUiMixin : public Base, public RenderDefaultUiObject
{
public:
    RenderDefaultUiMixin( const VisualObject& object )
        : Base( object ), RenderDefaultUiObject( object )
    {}
};

}
