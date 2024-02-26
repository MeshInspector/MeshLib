#include "MRMeshBoundarySelectionWidget.h"

#include "MRViewport.h"
#include "MRViewerInstance.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRObjectLines.h"

#include <MRMesh/MRRingIterator.h>
#include "MRMesh/MRPolyline.h"
#include "MRPickHoleBorderElement.h"
#include <MRMesh/MRObjectsAccess.h>

#include <MRMesh/MRSceneRoot.h>


namespace MR
{


void BoundarySelectionWidget::enable( bool isEnaled )
{
    isSelectorActive_ = isEnaled;
    if ( !isSelectorActive_ )
        holes_.clear();
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
    for ( auto& [obj, holes] : holes_ )
    {
        const auto hole = findClosestToMouseHoleEdge( getViewerInstance().mouseController().getMousePos(), obj, holes, mouseAccuracy_ );
        const int holeId = int( hole.holeIdx );
        if ( holeId != -1 )
            return std::make_pair( obj, hole );
    }
    return std::make_pair( nullptr, HoleEdgePoint() );
}

bool BoundarySelectionWidget::updateHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index, MR::Color color, float lineWidth )
{
    if ( object != nullptr )
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
    auto lineWidth = isSelectedAndHoveredTheSame_() ? std::max( params.hoveredLineWidth, params.ordinaryLineWidth ) : params.ordinaryLineWidth;
    updateHole_( selectedHoleObject_, selectedHoleIndex_, params.ordinaryColor, lineWidth );

    selectedHoleObject_ = object;
    selectedHoleIndex_ = index;

    lineWidth = isSelectedAndHoveredTheSame_() ? std::max( params.hoveredLineWidth, params.selectedLineWidth ) : params.selectedLineWidth;
    return updateHole_( selectedHoleObject_, selectedHoleIndex_, params.selectedColor, lineWidth );
}

bool BoundarySelectionWidget::hoverHole_( std::shared_ptr<MR::ObjectMeshHolder> object, int index )
{
    auto lineWidth = isSelectedAndHoveredTheSame_() ? params.selectedLineWidth : params.ordinaryLineWidth;
    updateHole_( hoveredHoleObject_, hoveredHoleIndex_, params.ordinaryColor, lineWidth );

    hoveredHoleObject_ = object;
    hoveredHoleIndex_ = index;

    lineWidth = isSelectedAndHoveredTheSame_() ? std::max( params.hoveredLineWidth, params.selectedLineWidth ) : params.hoveredLineWidth;
    return updateHole_( selectedHoleObject_, selectedHoleIndex_, params.selectedColor, lineWidth );
}

bool BoundarySelectionWidget::actionByPick_( ActionType actionType )
{

    const auto& [obj, hole] = getHoverdHole_();
    const int holeId = int( hole.holeIdx );

    if ( holeId == -1 )
        return false;

    auto objectHoles = holeLines_[obj];

    if ( actionType == ActionType::SelectHole )
        return selectHole( obj, holeId );
    else
        return hoverHole_( obj, holeId );

}

bool BoundarySelectionWidget::onMouseDown_( Viewer::MouseButton button, int modifier )
{
    if ( !isSelectorActive_ )
        return false;

    if ( button != Viewer::MouseButton::Left )
        return false;

    actionByPick_( ActionType::SelectHole );


    /*
    if ( modifier == GLFW_MOD_CONTROL )
    {
        const auto& [obj, hole] = getHoverdHole_();
        const int holeId = int( hole.holeIdx );

        if ( holeId == -1 )
            return false;

        auto ignoredHoles_ = holeLines_[obj];

        if ( ignoredHoles_[holeId].obj )
            ignoredHoles_[holeId].reset();
        else
        {
            ignoredHoles_[holeId].make( *obj_ );
            auto& obj = ignoredHoles_[holeId].obj;
            obj->setPolyline( getHoleBorder_( hole.edgePoint.e ) );
            obj->setName( "HoleBorder" );
            obj->setFrontColor( Color::purple(), false );
            obj->setLineWidth( ( float )lineWidth_ );
        }
    }
    else
    {
        auto e = findClosestToMouseHoleEdge( viewer->mouseController().getMousePos(), obj_, holes_, mouseAccuracy_ ).edgePoint.e;
        auto res = fillOnlyOneHole_( e );
        holes_ = obj_->mesh()->topology.findHoleRepresentiveEdges();
        ignoredHoles_ = std::vector<AncillaryLines>( holes_.size() );
        return res;
    }

    */
    return true;
}

bool BoundarySelectionWidget::onMouseMove_( int, int )
{
    if ( !isSelectorActive_ )
        return false;

    actionByPick_( ActionType::SelectHole );
}

AncillaryLines BoundarySelectionWidget::createAncillaryLines_( const std::shared_ptr <const ObjectMeshHolder> obj , MR::EdgeId hole)
{
    AncillaryLines al;
    al.make( *obj_ );
    auto& obj = ignoredHoles_[holeId].obj;
    obj->setPolyline( getHoleBorder_( hole.edgePoint.e ) );
    obj->setName( "HoleBorder" );
    obj->setFrontColor( Color::purple(), false );
    obj->setLineWidth( ( float )lineWidth_ );
}

void BoundarySelectionWidget::calculateHoles_()
{
    auto objects = getAllObjectsInTree<const ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for (auto& object : objects  )
    if ( isObjectValidToPick_( object ) )
    {
        auto holes = object->mesh()->topology.findHoleRepresentiveEdges();
        auto& polylines = holeLines_[object];
        polylines.reserve( holes.size() );
        for ( auto hole : holes )
            polylines.push_back( createAncillaryLines_( object , hole) );
            


    }

}

void BoundarySelectionWidget::create(
            BoundarySelectionWidgetCallBack onBoundarySelected,
            BoundarySelectionWidgetChecker isObjectValidToPick
)
{
    onBoundarySelected_ = std::move( onBoundarySelected );
    isObjectValidToPick_ = std::move( isObjectValidToPick );

    calculateHoles_();

    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}


void BoundarySelectionWidget::reset()
{
    enable( false );
    disconnect();
}
} // namespace MR 