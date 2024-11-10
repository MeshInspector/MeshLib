#pragma once

#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRHistoryStore.h"
#include "MRViewer/MRGladGlfw.h"
#include <unordered_map>
#include "MRViewer/MRPickHoleBorderElement.h"
#include "MRViewer/MRAncillaryLines.h"
#include "MRMesh/MRRingIterator.h"
#include "MRMesh/MRMesh.h"

namespace MR
{

// A widget that allows you to find, highlight and select boundaries (holes) in the mesh.
// To provide feedback during creation, it is necessary to specify a callback that will be called if a specific hole is selected.
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
    using PerObjectMeshChangedSignals = std::unordered_map < std::shared_ptr<MR::ObjectMeshHolder>, boost::signals2::scoped_connection>;


    // enable or disable widget
    MRVIEWER_API void enable( bool isEnabled );

    // create a widget and connect it. 
    // To create a widget, you need to provide 1 callbacks and one function that determines whether this object can be used to detect and select boundaries ( holes ).
    // All callback takes a shared pointer to an MR::ObjectMeshHolder as an argument.
    // onBoundarySelected: This callback is invoked when a boundary is selected.
    // isObjectValidToPick : Must return true or false. This callback is used to determine whether an object is valid for picking.
    MRVIEWER_API void create(
            BoundarySelectionWidgetCallBack onBoundarySelected,
            BoundarySelectionWidgetChecker isObjectValidToPick
    );

    // meshChangedSignal processor
    void onObjectChange_();

    // reset widget, clear internal variables and detach from signals.
    MRVIEWER_API void reset();

    // select one of the holes. Return true on success.
    MRVIEWER_API bool selectHole( std::shared_ptr<MR::ObjectMeshHolder> object, int index );

    // clear selection
    MRVIEWER_API void clear();

    // returns pair of selected hole ( in Edge representations) and objects on which particular hole is present
    MRVIEWER_API std::pair< std::shared_ptr<MR::ObjectMeshHolder>, EdgeId > getSelectHole() const;

    // collect and return vector of points ( verts coord ) for all edges in selected mesh boundary
    MRVIEWER_API std::vector<MR::Vector3f> getPointsForSelectedHole() const;

    // configuration params
    BoundarySelectionWidgetParams params;
private:

    float mouseAccuracy_{ 5.5f };

    bool isSelectorActive_ = false;

    PerObjectHoles holes_;
    PerObjectHolesPolylines holeLines_;
    PerObjectMeshChangedSignals  onMeshChangedSignals_;

    MRVIEWER_API bool onMouseDown_( MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    // create an ancillaryLines object (polyline) for given mesh hole, for visually preview it 
    AncillaryLines createAncillaryLines_( std::shared_ptr<ObjectMeshHolder>& obj, MR::EdgeId hole );

    // For given object and hole ( edge representation ), return a polyline around the hole boundary. 
    std::shared_ptr<MR::Polyline3> getHoleBorder_( const std::shared_ptr<ObjectMeshHolder> obj, EdgeId initEdge );

    // Currently it returns the first hole it comes across.
    // Those. if the condition is met for several holes( including holes on different objects ), then the first one available will be selected.
    std::pair<std::shared_ptr<MR::ObjectMeshHolder>, HoleEdgePoint> getHoverdHole_();

    // select hole
    bool selectHole_( std::shared_ptr<ObjectMeshHolder> object, int index, bool writeHistory = true );

    // update color for one of the polylines
    bool updateHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index, MR::Color color, float lineWidth );


    enum class ActionType {
        SelectHole,
        HoverHole
    };

    // pick processor, for both mouseDown and MouseMove events.
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

    // return is selected hole hovered at this time or not. 
    bool isSelectedAndHoveredTheSame_();

    // hover particular hole/
    bool hoverHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index );

    // calculate and store all holes on meshes which allowed by isObjectValidToPick_ callback
    void calculateHoles_();

    friend class ChangeBoundarySelectionHistoryAction;
};

class ChangeBoundarySelectionHistoryAction : public HistoryAction
{
public:
    ChangeBoundarySelectionHistoryAction( std::string name, BoundarySelectionWidget& widget, std::shared_ptr<ObjectMeshHolder> object, int index );

public:
    // HistoryAction
    [[nodiscard]] std::string name() const override { return name_; }

    void action( Type type ) override;

    [[nodiscard]] size_t heapBytes() const override;

private:
    std::string name_;
    BoundarySelectionWidget& widget_;
    std::shared_ptr<ObjectMeshHolder> prevSelectedHoleObject_;
    std::shared_ptr<ObjectMeshHolder> nextSelectedHoleObject_;
    int prevSelectedHoleIndex_;
    int nextSelectedHoleIndex_;
};

} // namespace MR
