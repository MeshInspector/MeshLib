#include "MRNameTagClickListener.h"

#include "MRViewer/MRViewer.h"

namespace MR
{

void NameTagClickListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->getMenuPluginAs<RibbonMenu>()->nameTagClickSignal.connect( group, MAKE_SLOT( &NameTagClickListener::onNameTagClicked_ ), pos );
}

}
