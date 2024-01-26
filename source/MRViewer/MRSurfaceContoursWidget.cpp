#include "MRSurfaceContoursWidget.h"
#include "MRViewport.h"
#include "MRMesh/MRHistoryAction.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRObjectMesh.h"

namespace MR
{

void updateBaseColor( std::shared_ptr<SurfacePointWidget> point, const Color& color )
{
    auto params = point->getParameters();
    params.baseColor = color;
    point->setParameters( params );
}

// History classes;

std::string AddPointActionPickerPoint::name() const
{
    return "Add Point " + widget_.params.historySpecification;
}

void AddPointActionPickerPoint::action( Type actionType )
{
    if ( !widget_.isPickerActive_ )
        return;

    auto& contour = widget_.pickedPoints_[obj_];
    if ( actionType == Type::Undo )
    {
        contour.pop_back();

        if ( !contour.empty() )
            updateBaseColor( contour.back(), widget_.params.lastPoitColor );

        widget_.activeIndex_ = int( contour.size() );
        widget_.activeObject_ = obj_;

        widget_.onPointRemove_( obj_ );
    }
    else
    {
        if ( !contour.empty() )
            updateBaseColor( contour.back(), widget_.params.ordinaryPointColor );

        contour.push_back( widget_.createPickWidget_( obj_, point_ ) );

        updateBaseColor( contour.back(), widget_.params.lastPoitColor );


        widget_.onPointAdd_( obj_ );
    }

}
size_t AddPointActionPickerPoint::heapBytes() const
{
    return 0; //this undo action will be deleted in widget disable
}



std::string RemovePointActionPickerPoint::name() const
{
    return "Remove Point " + widget_.params.historySpecification;
}
void RemovePointActionPickerPoint::action( Type actionType )
{
    if ( !widget_.isPickerActive_ )
        return;
    auto& contour = widget_.pickedPoints_[obj_];
    if ( actionType == Type::Undo )
    {
        if ( index_ == contour.size() && !contour.empty() )
            updateBaseColor( contour.back(), widget_.params.ordinaryPointColor );

        contour.insert( contour.begin() + index_, widget_.createPickWidget_( obj_, point_ ) );

        if ( index_ + 1 == contour.size() )
            updateBaseColor( contour.back(), widget_.params.lastPoitColor );
        widget_.activeIndex_ = index_;
        widget_.activeObject_ = obj_;

        widget_.onPointAdd_( obj_ );
        contour.back()->setHovered( false );

    }
    else
    {
        contour.erase( contour.begin() + index_ );

        if ( index_ == contour.size() && !contour.empty() )
            updateBaseColor( contour.back(), widget_.params.lastPoitColor );

        widget_.activeIndex_ = index_;
        widget_.activeObject_ = obj_;

        widget_.onPointRemove_( obj_ );
    }
}

size_t RemovePointActionPickerPoint::heapBytes() const
{
    return 0; //this undo action will be deleted in widget disable
}

std::string ChangePointActionPickerPoint::name() const
{
    return "Move Point " + widget_.params.historySpecification;
}
void ChangePointActionPickerPoint::action( Type )
{
    if ( !widget_.isPickerActive_ )
        return;

    widget_.pickedPoints_[obj_][index_]->updateCurrentPosition( point_ );
    widget_.activeIndex_ = index_;
    widget_.activeObject_ = obj_;
    widget_.onPointMoveFinish_( obj_ );
}

size_t ChangePointActionPickerPoint::heapBytes() const
{
    return 0; //this undo action will be deleted in widget disable
}

void SurfaceContoursWidget::enable( bool isEnaled )
{
    isPickerActive_ = isEnaled;
    if ( !isPickerActive_ )
        pickedPoints_.clear();
}

std::shared_ptr<SurfacePointWidget> SurfaceContoursWidget::createPickWidget_( const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& pt )
{
    auto newPoint = std::make_shared<SurfacePointWidget>();
    newPoint->setAutoHover( false );
    newPoint->setParameters( params.surfacePointParams );
    auto objMesh = std::dynamic_pointer_cast< MR::ObjectMesh > ( obj );
    if ( objMesh )
        newPoint->create( objMesh, pt );
    else
    {
        assert( 1 > 2 ); // due to surfacePointWidget unable to work with non ObjectMesh objects.
        return {};
    }
    std::weak_ptr<SurfacePointWidget> curentPoint = newPoint;
    newPoint->setStartMoveCallback( [this, obj, curentPoint] ( const MeshTriPoint& point )
    {
        const bool closedPath = isClosedCountour( obj );

        if ( closedPath )
        {
            const auto& contour = pickedPoints_[obj];
            if ( curentPoint.lock() == contour[0] )
            {
                if ( params.writeHistory )
                {
                    SCOPED_HISTORY( "Move Point " + params.historySpecification );
                    AppendHistory<ChangePointActionPickerPoint>( *this, obj, point, activeIndex_ );
                    AppendHistory<ChangePointActionPickerPoint>( *this, obj, point, int( contour.size() ) - 1 );
                }
                moveClosedPoint_ = true;
            }
            else
            {
                if ( params.writeHistory )
                    AppendHistory<ChangePointActionPickerPoint>( *this, obj, point, activeIndex_ );
            }
        }
        else
        {
            if ( params.writeHistory )
                AppendHistory<ChangePointActionPickerPoint>( *this, obj, point, activeIndex_ );
        }
        activeChange_ = true;
        onPointMove_( obj );

    } );
    newPoint->setEndMoveCallback( [this, obj, curentPoint] ( const MeshTriPoint& point )
    {
        if ( moveClosedPoint_ )
        {
            const auto& contour = pickedPoints_[obj];
            if ( curentPoint.lock() == contour[0] )
            {
                contour.back()->updateCurrentPosition( point );
                moveClosedPoint_ = false;
            }
        }
        activeChange_ = false;
        onPointMoveFinish_( obj );
    } );

    return newPoint;
}

bool SurfaceContoursWidget::isClosedCountour( const std::shared_ptr<ObjectMeshHolder>& obj )
{
    auto pointsIt = pickedPoints_.find( obj );
    if ( pointsIt == pickedPoints_.end() )
        return false;
    return pointsIt->second.size() > 1 && pointsIt->second[0]->getCurrentPosition() == pointsIt->second.back()->getCurrentPosition();
}

void SurfaceContoursWidget::updateAllPointsWidgetParams( const SurfaceContoursWidgetParams& p )
{
    const auto& oldParams = params;

    for ( auto& [parentObj, contour] : pickedPoints_ )
        for ( auto& point : contour )
        {
            auto pointParams = point->getParameters();
            point->setParameters( p.surfacePointParams );

            if ( pointParams.baseColor == oldParams.ordinaryPointColor )
                updateBaseColor( point, p.ordinaryPointColor );
            else if ( pointParams.baseColor == oldParams.lastPoitColor )
                updateBaseColor( point, p.lastPoitColor );
            else if ( pointParams.baseColor == oldParams.closeContourPointColor )
                updateBaseColor( point, p.closeContourPointColor );
        }

    params = p;
}

std::pair<std::shared_ptr<MR::ObjectMeshHolder>, int> SurfaceContoursWidget::getActivePoint()
{
    return std::pair<std::shared_ptr<MR::ObjectMeshHolder>, int>( activeObject_, activeIndex_ );
}

void SurfaceContoursWidget::setActivePoint( std::shared_ptr<MR::ObjectMeshHolder> obj, int index )
{
    assert( pickedPoints_[obj].size() < index );

    updateBaseColor( pickedPoints_[obj][index], params.lastPoitColor );
    updateBaseColor( pickedPoints_[activeObject_][activeIndex_], params.ordinaryPointColor );

    activeIndex_ = index;
    activeObject_ = obj;
}

bool SurfaceContoursWidget::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( !isPickerActive_ )
        return false;

    if ( button != Viewer::MouseButton::Left )
        return false;

    auto [obj, pick] = getViewerInstance().viewport().pick_render_object();

    auto addPoint = [this] ( const std::shared_ptr<ObjectMeshHolder> obj, const MeshTriPoint& triPoint, bool close )
    {
        if ( !isObjectValidToPick_( obj ) )
            return;

        auto& contour = pickedPoints_[obj];
        if ( !contour.empty() )
            updateBaseColor( contour.back(), params.ordinaryPointColor );

        if ( params.writeHistory )
            AppendHistory<AddPointActionPickerPoint>( *this, obj, triPoint );

        contour.push_back( createPickWidget_( obj, triPoint ) );
        updateBaseColor( contour.back(), close ? params.closeContourPointColor : params.lastPoitColor );

        activeIndex_ = static_cast< int >( contour.size() - 1 );
        activeObject_ = obj;

        onPointAdd_( obj );
    };
    auto removePoint = [this] ( const std::shared_ptr<ObjectMeshHolder> obj, int pickedIndex )
    {
        auto& contour = pickedPoints_[obj];

        if ( pickedIndex == int( contour.size() ) - 1 && contour.size() > 1 )
        {
            updateBaseColor( contour[pickedIndex - 1], params.lastPoitColor );
        }

        if ( params.writeHistory )
            AppendHistory<RemovePointActionPickerPoint>( *this, obj, contour[pickedIndex]->getCurrentPosition(), pickedIndex );

        contour.erase( contour.begin() + pickedIndex );
        activeIndex_ = pickedIndex;
        activeObject_ = obj;
        onPointRemove_( obj );
    };

    if ( !mod ) // just add new point 
    {
        auto objMesh = std::dynamic_pointer_cast< ObjectMeshHolder >( obj );
        if ( !objMesh )
            return false;

        // all pick in point (without mod) must not comes here. Point widget must "eat" them. 
        if ( isClosedCountour( objMesh ) )
            return false;

        assert( objMesh != nullptr ); // contoursWidget_ can join for mesh objects only

        auto triPoint = objMesh->mesh()->toTriPoint( pick );
        addPoint( objMesh, triPoint, false );
        return true;
    }
    else if ( mod == params.widgetContourCloseMod ) // close contour case 
    {
        // Try to find parent object 
        auto isFirstPointOnCountourClicked = false;
        std::shared_ptr<ObjectMeshHolder> objectToCloseCoutour = nullptr;
        for ( const auto& [parentObj, contour] : pickedPoints_ )
        {
            if ( contour.size() > 2 && obj == contour[0]->getPickSphere() )
            {
                isFirstPointOnCountourClicked = true;
                objectToCloseCoutour = parentObj;
                break;
            }
        }
        if ( !isFirstPointOnCountourClicked )
            return false;

        if ( isClosedCountour( objectToCloseCoutour ) )
            return false;

        assert( objectToCloseCoutour != nullptr );
        auto triPoint = pickedPoints_[objectToCloseCoutour][0]->getCurrentPosition();
        addPoint( objectToCloseCoutour, triPoint, true );
        activeIndex_ = 0;
        return true;
    }
    else if ( mod == params.widgetDeletePointMod )  // remove point case 
    {
        if ( pickedPoints_.empty() )
            return false;

        int pickedIndex = -1;
        std::shared_ptr<MR::ObjectMeshHolder> pickedObj = nullptr;

        // try to find point to remove 
        for ( const auto& [parentObj, contour] : pickedPoints_ )
            for ( int i = 0; i < contour.size(); ++i )
            {
                const auto& point = contour[i];
                if ( obj == point->getPickSphere() )
                {
                    pickedIndex = i;
                    pickedObj = parentObj;
                    break;
                }

            }

        if ( ( pickedIndex == -1 ) || ( pickedObj == nullptr ) )
            return false;

        if ( isClosedCountour( pickedObj ) )
        {
            assert( pickedIndex >= 0 );
            assert( pickedObj != nullptr );

            auto& contour = pickedPoints_[pickedObj];
            assert( pickedIndex != contour.size() - 1 ); // unable to pick point which is close countour

            if ( params.writeHistory )
                SCOPED_HISTORY( "Remove Point" + params.historySpecification );

            // 4 points - minimal non-trivial closed path
            // last on is a "pseudo" point to close contour
            // so remove last point which is close contour in case of: 
            // 1) Contour can no longer been closed (only 2 valid point left ) 
            // 2) First point need to be remove. Last point will be restored later.  
            if ( contour.size() == 4 || pickedIndex == 0 )
                removePoint( pickedObj, ( int )contour.size() - 1 );

            // Remove point marked to be removed. 
            removePoint( pickedObj, pickedIndex );

            // Restore close countour. 
            if ( contour.size() > 2 && pickedIndex == 0 )
                addPoint( pickedObj, contour[0]->getCurrentPosition(), true );
        }
        else
            removePoint( pickedObj, pickedIndex );
    }
    return false;
}

bool SurfaceContoursWidget::onMouseMove_( int, int )
{
    if ( !isPickerActive_ )
        return false;

    if ( pickedPoints_.empty() || activeChange_ )
        return false;

    auto [obj, pick] = getViewerInstance().viewport().pick_render_object();
    if ( !obj )
        return false;

    for ( auto contour : pickedPoints_ )
        for ( int i = 0; i < contour.second.size(); ++i )
        {
            const auto& point = contour.second[i];
            bool hovered = obj == point->getPickSphere();
            point->setHovered( hovered );
            if ( hovered )
            {
                activeIndex_ = i;
                activeObject_ = contour.first;
            }
        }
    return false;
}

void SurfaceContoursWidget::create(
        PickerPointCallBack onPointAdd,
        PickerPointCallBack onPointMove,
        PickerPointCallBack onPointMoveFinish,
        PickerPointCallBack onPointRemove,
        PickerPointObjectChecker isObjectValidToPick
)
{
    onPointAdd_ = std::move( onPointAdd );
    onPointMove_ = std::move( onPointMove );
    onPointMoveFinish_ = std::move( onPointMoveFinish );
    onPointRemove_ = std::move( onPointRemove );
    isObjectValidToPick_ = std::move( isObjectValidToPick );

    clear();

    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void SurfaceContoursWidget::clear()
{
    SCOPED_HISTORY( "Remove All Point" + params.historySpecification );
    for ( auto& [obj, contour] : pickedPoints_ )
        for ( int i = static_cast< int >( contour.size() - 1 ); i >= 0; --i )        
            AppendHistory<RemovePointActionPickerPoint>( *this, obj, contour[i]->getCurrentPosition(), i );  
    pickedPoints_.clear();
    activeIndex_ = 0;
    activeObject_ = nullptr;
}

void SurfaceContoursWidget::reset()
{
    clear();
    enable( false );

    if ( ( params.writeHistory ) && ( params.filterHistoryonReset ) )
    {
        FilterHistoryByCondition( [&] ( const std::shared_ptr<HistoryAction>& action )
        {
            bool res =
                bool( std::dynamic_pointer_cast< AddPointActionPickerPoint >( action ) ) ||
                bool( std::dynamic_pointer_cast< RemovePointActionPickerPoint >( action ) ) ||
                bool( std::dynamic_pointer_cast< ChangePointActionPickerPoint >( action ) );
            return res;
        } );
    }

    disconnect();
}


} // namespace MR 