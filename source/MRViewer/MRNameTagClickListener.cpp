#include "MRNameTagClickListener.h"

#include "MRViewer/MRViewer.h"

namespace MR
{

void NameTagClickListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    auto menu = viewer->getMenuPlugin();
    assert( menu );
    if ( menu )
        connection_ = menu->nameTagClickSignal.connect( group, MAKE_SLOT( &NameTagClickListener::onNameTagClicked_ ), pos );
}

}
