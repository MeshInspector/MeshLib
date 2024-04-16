#pragma once

#include "exports.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewerEventsListener.h"

namespace MR
{

// A helper base class to subscribe to `RibbonMenu::manuallySelectObjectSignal`.
struct MRVIEWER_CLASS NameTagClickListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( NameTagClickListener );
    virtual ~NameTagClickListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onNameTagClicked_( Object& object, RibbonMenu::NameTagSelectionMode mode ) = 0;
};

}
