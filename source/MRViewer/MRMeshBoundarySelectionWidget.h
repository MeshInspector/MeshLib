#pragma once

#include "MRViewer.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRHistoryStore.h"
#include "MRViewer/MRGladGlfw.h"
#include <unordered_map>
#include "MRViewer/MRPickHoleBorderElement.h"


namespace MR
{



/// Helper class to manage ancillary visual lines used by plugins
struct AncillaryLines
{
    std::shared_ptr<ObjectLines> obj;

    AncillaryLines() = default;

    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryLines( AncillaryLines&& b ) noexcept : obj{ std::move( b.obj ) }
    {}
    AncillaryLines& operator =( AncillaryLines&& b )
    {
        reset(); obj = std::move( b.obj ); return *this;
    }

    /// Make not-pickable ancillary object, link it to parent object, and set line geometry
    explicit AncillaryLines( Object& parent, const Contours3f& contours = {} )
    {
        make( parent, contours );
    }

    /// Make not-pickable ancillary object, link it to parent object, and set line geometry
    void make( Object& parent, const Contours3f& contours = {} );

    /// detach owned object from parent, stops owning it
    void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryLines()
    {
        reset();
    }

    /// Set line geometry
    void setContours( const Contours3f& contours );

    /// Reset line geometry
    void resetContours();

    /// Set depth test
    void setDepthTest( bool depthTest );
};


class MRVIEWER_CLASS BoundarySelectionWidget : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:

    struct BoundarySelectionWidgetParams {

        MR::Color ordinaryColor = MR::Color::gray();
        float ordinaryLineWidth = 3;

        MR::Color hoveredColor = MR::Color::green();
        float hoveredLineWidth = 4;

        MR::Color selectedColor = MR::Color::purple();
        float selectedLineWidth = 3;

    };

    using BoundarySelectionWidgetCallBack = std::function<void( std::shared_ptr<const MR::ObjectMeshHolder> )>;
    using BoundarySelectionWidgetChecker = std::function<bool( std::shared_ptr<const MR::ObjectMeshHolder> )>;

    using HolesOnObject = std::vector<MR::EdgeId>;
    using PerObjectHoles = std::unordered_map <std::shared_ptr<MR::ObjectMeshHolder>, HolesOnObject>;
    using PerObjectHolesPolylines = std::unordered_map <std::shared_ptr<MR::ObjectMeshHolder>, std::vector<AncillaryLines>>;


    // enable or disable widget
    MRVIEWER_API void enable( bool isEnaled );

    // create a widget and connect it. 
    // To create a widget, you need to provide 1 callbacks and one function that determines whether this object can be used to place points.
    // All callback takes a shared pointer to an MR::ObjectMeshHolder as an argument.
    // onBoundarySelected: This callback is invoked when a boundary is selected.
    // isObjectValidToPick : Must returh true or false. This callback is used to determine whether an object is valid for picking.
    MRVIEWER_API void create(
            BoundarySelectionWidgetCallBack onBoundarySelected,
            BoundarySelectionWidgetChecker isObjectValidToPick
    );

    // reset widget, clear internal variables and detach from signals.
    MRVIEWER_API void reset();

    // select one of the holes. Return succsess.
    bool selectHole( std::shared_ptr<MR::ObjectMeshHolder> object, int index );



    // configuration params
    BoundarySelectionWidgetParams params;
private:

    float lineWidth_ = 3.0f;

    float mouseAccuracy_{ 5.5f };

    bool isSelectorActive_ = false;

    PerObjectHoles holes_;
    PerObjectHolesPolylines holeLines_;

    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    AncillaryLines createAncillaryLines_( std::shared_ptr<ObjectMeshHolder>& obj, MR::EdgeId hole );


    // For given object and hole ( edge representation ), return a polyline around the hole boundary. 
    std::shared_ptr<MR::Polyline3> getHoleBorder_( const std::shared_ptr<ObjectMeshHolder> obj, EdgeId initEdge );

    // Currently it returns the first hole it comes across.
    // Those. if the condition is met for several holes( including holes on different objects ), then the first one available will be selected.
    std::pair<std::shared_ptr<MR::ObjectMeshHolder>, HoleEdgePoint> getHoverdHole_();

    // update color for one of the polylines
    bool updateHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index, MR::Color color, float lineWidth );


    enum class ActionType {
        SelectHole,
        HoverHole
    };

    bool actionByPick_( ActionType actionType );


    // CallBack functions
    BoundarySelectionWidgetCallBack onBoundarySelected_;
    BoundarySelectionWidgetChecker isObjectValidToPick_;

    // selected hole
    std::shared_ptr<MR::ObjectMeshHolder> selectedHoleObject_;
    int selectedHoleIndex_;

    // hovered hole 
    std::shared_ptr<MR::ObjectMeshHolder> hoveredHoleObject_;
    int hoveredHoleIndex_;

    bool isSelectedAndHoveredTheSame_();

    bool hoverHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index );

    void calculateHoles_();

};

}