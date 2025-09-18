#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRViewer/MRViewportGlobalBasis.h"
#include "MRCommonPlugins/Basic/MRDrawViewportWidgetsItem.h"

namespace MR
{
class ShowGlobalBasisMenuItem : public RibbonMenuItem, public ProvidesViewportWidget
{
public:
    ShowGlobalBasisMenuItem();

    enum class State
    {
        // Those define the cycle order when clicking the button.
        none,
        basis,
        basisAndGrid,
        _count,
    };

    [[nodiscard]] static State nextState( State state )
    {
        state = State( int( state ) + 1 );
        if ( state == State::_count )
            state = State{};
        return state;
    }

    [[nodiscard]] State getCurrentState( ViewportId id )
    {
        auto& viewer = Viewer::instanceRef();
        bool isVisible = viewer.globalBasis->isVisible( id );
        bool isGridVisible = viewer.globalBasis->isGridVisible( id );

        if ( !isVisible )
            return State::none;
        else if ( !isGridVisible )
            return State::basis;
        else
            return State::basisAndGrid;
    }

    void setState( ViewportId id, State state )
    {
        auto& viewer = Viewer::instanceRef();
        switch ( state )
        {
        case State::none:
            viewer.globalBasis->setVisible( false, id );
            viewer.globalBasis->setGridVisible( false, id );
            break;
        case State::basis:
            viewer.globalBasis->setVisible( true, id );
            viewer.globalBasis->setGridVisible( false, id );
            break;
        case State::basisAndGrid:
            viewer.globalBasis->setVisible( true, id );
            viewer.globalBasis->setGridVisible( true, id );
            break;
        case State::_count:
            // Should be unreachable.
            break;
        }
    }

    void cycleState( ViewportId id )
    {
        setState( id, nextState( getCurrentState( id ) ) );
    }

    bool action() override
    {
        cycleState( getViewerInstance().viewport().id );
        return false;
    }

    void providedViewportWidgets( ViewportWidgetInterface& in ) override
    {
        auto id = in.viewportId();
        if ( !showButtonInViewports.contains( id ) )
            return;

        State state = getCurrentState( id );

        const char* icon = nullptr;
        switch ( state )
        {
        case State::none:
            icon = "Viewport basis off";
            break;
        case State::basis:
            icon = "Viewport basis";
            break;
        case State::basisAndGrid:
            icon = "Viewport basis and grid";
            break;
        case State::_count:
            // Should be unreachable.
            break;
        }

        state = nextState( state );

        const char* tooltip = nullptr;
        switch ( state )
        {
        case State::none:
            tooltip = "Basis: hide basis and grid";
            break;
        case State::basis:
            tooltip = "Basis: show basis";
            break;
        case State::basisAndGrid:
            tooltip = "Basis: show both basis and grid";
            break;
        case State::_count:
            // Should be unreachable.
            break;
        }

        in.addButton( 20, "Basis", false, icon, tooltip,
            [this, id, state]{ setState( id, state ); }
        );
    }

    ViewportMask showButtonInViewports = ViewportMask::all();
};

ShowGlobalBasisMenuItem::ShowGlobalBasisMenuItem() :
    RibbonMenuItem( "Show_Hide Global Basis" )
{}

MR_REGISTER_RIBBON_ITEM( ShowGlobalBasisMenuItem )
}
