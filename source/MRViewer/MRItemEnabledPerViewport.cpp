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

}
