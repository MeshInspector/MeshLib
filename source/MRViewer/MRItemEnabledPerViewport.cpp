#include "MRItemEnabledPerViewport.h"

namespace MR
{

ViewportMask ItemEnabledPerViewport::getEnabledViewports() const
{
    return enabledViewports_;
}

void ItemEnabledPerViewport::setEnabledViewports( ViewportMask newMask )
{
    enabledViewports_ = newMask;
}

ItemEnabledPerViewport::ItemEnabledPerViewport()
    // This base class initializer will never be used, because the base is virtual.
    // All virtual bases are always initialized by the most-derived class.
    : RibbonMenuItem( "??" )
{}

}
