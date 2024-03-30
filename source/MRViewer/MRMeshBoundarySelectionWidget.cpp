#include "MRMeshBoundarySelectionWidget.h"

#include "MRViewport.h"
#include "MRViewerInstance.h"
#include "MRPickHoleBorderElement.h"

#include "MRViewer/MRMouseController.h"
#include "MRViewer/MRAncillaryLines.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"

#include <MRMesh/MRRingIterator.h>
#include "MRMesh/MRPolyline.h"
#include <MRMesh/MRObjectsAccess.h>
#include <MRMesh/MRSceneRoot.h>


namespace MR
{

void BoundarySelectionWidget::enable( bool isEnaled )
{
    isSelectorActive_ = isEnaled;
    if ( !isSelectorActive_ )
    {
        onMeshChangedSignals_.clear();
        holes_.clear();
        holeLines_.clear();
        selectedHoleObject_ = nullptr;
        selectedHoleIndex_ = -1;
        hoveredHoleIndex_ = -1;
        hoveredHoleObject_ = nullptr;
    }
    else
    {
        calculateHoles_();
        selectedHoleObject_ = nullptr;
        selectedHoleIndex_ = -1;
        hoveredHoleIndex_ = -1;
        hoveredHoleObject_ = nullptr;
    }
}

std::shared_ptr<MR::Polyline3> BoundarySelectionWidget::getHoleBorder_( const std::shared_ptr<ObjectMeshHolder> obj, EdgeId initEdge )
{
    if ( !initEdge.valid() )
        return {};

    EdgePath path;
    const auto& mesh = *obj->mesh();
    for ( auto e : leftRing( mesh.topology, initEdge ) )
    {
        path.push_back( e );
    }
    std::shared_ptr<Polyline3> polyline = std::make_shared<Polyline3>();
    if ( !path.empty() )
        polyline->addFromEdgePath( mesh, path );

    return polyline;
}

std::pair  <std::shared_ptr<MR::ObjectMeshHolder>, HoleEdgePoint> BoundarySelectionWidget::getHoverdHole_()
{
    const auto& mousePos = getViewerInstance().mouseController().getMousePos();
    for ( auto& [obj, holes] : holes_ )
    {
        const auto hole = findClosestToMouseHoleEdge( mousePos, obj, holes, mouseAccuracy_ );
        const int holeId = int( hole.holeIdx );
        if ( holeId != -1 )
            return std::make_pair( obj, hole );
    }
    return std::make_pair( nullptr, HoleEdgePoint() );
}

bool BoundarySelectionWidget::updateHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index, MR::Color color, float lineWidth )
{
    if ( ( object != nullptr ) || ( index < 0 ) )
    {
        auto& objectPolylines = holeLines_[object];
        if ( objectPolylines.size() > index )

        {
            auto& polyline = objectPolylines[index].obj;
            polyline->setFrontColor( color, false );
            polyline->setLineWidth( lineWidth );
            return true;
        }
    }
    return false;
}

bool  BoundarySelectionWidget::isSelectedAndHoveredTheSame_()
{
    return  ( ( selectedHoleObject_ == hoveredHoleObject_ ) && ( selectedHoleIndex_ == hoveredHoleIndex_ ) );
}

bool BoundarySelectionWidget::selectHole( std::shared_ptr<MR::ObjectMeshHolder> object, int index )
{
    updateHole_( selectedHoleObject_, selectedHoleIndex_, params.ordinaryColor, params.ordinaryLineWidth );

    selectedHoleObject_ = object;
    selectedHoleIndex_ = index;

    auto lineWidth = isSelectedAndHoveredTheSame_() ? std::max( params.hoveredLineWidth, params.selectedLineWidth ) : params.selectedLineWidth;
    auto result = updateHole_( selectedHoleObject_, selectedHoleIndex_, params.selectedColor, lineWidth );
    if ( index >= 0 )
        onBoundarySelected_( object );
    return result;
}

bool BoundarySelectionWidget::hoverHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index )
{
    if ( ( hoveredHoleObject_ == object ) && ( index == hoveredHoleIndex_ ) )
        return false;

    float  lineWidth;
    MR::Color lineColor;

    if ( isSelectedAndHoveredTheSame_() )
    {
        lineWidth = params.selectedLineWidth;
        lineColor = params.selectedColor;
    }
    else
    {
        lineColor = params.ordinaryColor;
        lineWidth = params.ordinaryLineWidth;
    }
    updateHole_( hoveredHoleObject_, hoveredHoleIndex_, lineColor, lineWidth );

    hoveredHoleObject_ = object;
    hoveredHoleIndex_ = index;

    if ( isSelectedAndHoveredTheSame_() )
    {
        lineWidth = std::max( params.hoveredLineWidth, params.selectedLineWidth );
        lineColor = params.selectedColor;
    }
    else
    {
        lineWidth = params.hoveredLineWidth;
        lineColor = params.hoveredColor;
    }
    return updateHole_( hoveredHoleObject_, hoveredHoleIndex_, lineColor, lineWidth );
}

std::pair< std::shared_ptr<MR::ObjectMeshHolder>, EdgeId > BoundarySelectionWidget::getSelectHole() const
{
    EdgeId hole = holes_.at( selectedHoleObject_ )[selectedHoleIndex_];
    return std::make_pair( selectedHoleObject_, hole );
}

std::vector<MR::Vector3f> BoundarySelectionWidget::getPointsForSelectedHole() const
{
    if ( holes_.count( selectedHoleObject_ ) == 0 )
        return {};

    std::vector<MR::Vector3f> result;
    EdgeId hole = holes_.at( selectedHoleObject_ )[selectedHoleIndex_];
    auto& mesh = *selectedHoleObject_->mesh();
    for ( auto e : leftRing( mesh.topology, hole ) )
    {
        auto v = mesh.topology.org( e );
        result.push_back( mesh.points[v] );
    }

    return result;
}

bool BoundarySelectionWidget::actionByPick_( ActionType actionType )
{

    const auto& [obj, hole] = getHoverdHole_();
    const int holeId = int( hole.holeIdx );



    if ( actionType == ActionType::SelectHole )
    {
        return selectHole( obj, holeId );
    }
    else
    {
        return hoverHole_( obj, holeId );
    }


}

bool BoundarySelectionWidget::onMouseDown_( Viewer::MouseButton button, int /*modifier*/ )
{
    if ( !isSelectorActive_ )
        return false;

    if ( button != Viewer::MouseButton::Left )
        return false;

    actionByPick_( ActionType::SelectHole );

    return true;
}

bool BoundarySelectionWidget::onMouseMove_( int, int )
{
    if ( !isSelectorActive_ )
        return false;

    return actionByPick_( ActionType::HoverHole );
}

AncillaryLines BoundarySelectionWidget::createAncillaryLines_( std::shared_ptr <ObjectMeshHolder>& rootObj, MR::EdgeId hole )
{
    AncillaryLines al;
    al.make( *rootObj );
    auto& object = al.obj;
    object->setPolyline( getHoleBorder_( rootObj, hole ) );
    object->setName( "HoleBorder" );
    object->setFrontColor( params.ordinaryColor, false );
    object->setLineWidth( params.ordinaryLineWidth );
    return al;
}

void BoundarySelectionWidget::calculateHoles_()
{
    auto objects = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( auto& object : objects )
        if ( isObjectValidToPick_( object ) )
        {
            //  join widget to objects meshChangedSignal 
            auto oMesh = std::dynamic_pointer_cast< ObjectMesh > ( object );
            if ( oMesh )
            {
                onMeshChangedSignals_[object] = oMesh->meshChangedSignal.connect( [this] ( uint32_t )
                {
                    onObjectChange_();
                } );
            }

            // calculate holes 
            auto& holes = holes_[object];
            auto& polylines = holeLines_[object];

            holes = object->mesh()->topology.findHoleRepresentiveEdges();
            polylines.reserve( holes.size() );
            for ( auto hole : holes )
                polylines.push_back( createAncillaryLines_( object, hole ) );


        }
}

void BoundarySelectionWidget::create(
            BoundarySelectionWidgetCallBack onBoundarySelected,
            BoundarySelectionWidgetChecker isObjectValidToPick
)
{
    onBoundarySelected_ = std::move( onBoundarySelected );
    isObjectValidToPick_ = std::move( isObjectValidToPick );

    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void BoundarySelectionWidget::onObjectChange_()
{
    if ( isSelectorActive_ )
    {
        // reset and recalculate all holes
        enable( false );
        enable( true );
    }
}

void BoundarySelectionWidget::reset()
{
    enable( false );
    disconnect();
}
} // namespace MR 