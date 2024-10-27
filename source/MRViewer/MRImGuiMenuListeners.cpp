#include "MRImGuiMenuListeners.h"
#include "MRMakeSlot.h"

namespace MR
{

void NameTagClickListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    auto menu = ImGuiMenu::instance();
    assert( menu );
    if ( menu )
        connection_ = menu->nameTagClickSignal.connect( group, MAKE_SLOT( &NameTagClickListener::onNameTagClicked_ ), pos );
}

void DrawSceneUiListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    auto menu = ImGuiMenu::instance();
    assert( menu );
    if ( menu )
        connection_ = menu->drawSceneUiSignal.connect( group, MAKE_SLOT( &DrawSceneUiListener::onDrawSceneUi_ ), pos );
}

}
