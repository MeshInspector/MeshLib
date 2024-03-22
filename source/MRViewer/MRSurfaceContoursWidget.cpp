#include "MRSurfaceContoursWidget.h"
#include "MRViewport.h"
#include "MRMesh/MRHistoryAction.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRPointOnObject.h"

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

        widget_.highlightLastPoint( obj_ );
        widget_.activeIndex_ = int( contour.size() - 1 );
        widget_.activeObject_ = obj_;

        widget_.onPointRemove_( obj_ );
    }
    else
    {
        contour.push_back( widget_.createPickWidget_( obj_, point_ ) );

        widget_.activeIndex_ = int( contour.size() - 1 );
        widget_.activeObject_ = obj_;
        widget_.highlightLastPoint( obj_ );

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
        contour.insert( contour.begin() + index_, widget_.createPickWidget_( obj_, point_ ) );

        widget_.activeIndex_ = index_;
        widget_.activeObject_ = obj_;
        widget_.highlightLastPoint( obj_ );
        widget_.onPointAdd_( obj_ );
        contour.back()->setHovered( false );

    }
    else
    {
        contour.erase( contour.begin() + index_ );

        widget_.activeIndex_ = index_;
        widget_.activeObject_ = obj_;
        widget_.highlightLastPoint( obj_ );
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



std::shared_ptr<SurfacePointWidget> SurfaceContoursWidget::createPickWidget_( const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& pt )
{
    auto newPoint = std::make_shared<SurfacePointWidget>();
    newPoint->setAutoHover( false );
    newPoint->setParameters( params.surfacePointParams );
    newPoint->create( obj, pt );

    std::weak_ptr<SurfacePointWidget> curentPoint = newPoint;
    newPoint->setStartMoveCallback( [this, obj, curentPoint] ( const PickedPoint& point )
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
    newPoint->setEndMoveCallback( [this, obj, curentPoint] ( const PickedPoint& point )
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

bool SurfaceContoursWidget::isClosedCountour( const std::shared_ptr<VisualObject>& obj )
{
    auto pointsIt = pickedPoints_.find( obj );
    if ( pointsIt == pickedPoints_.end() )
        return false;
    return pointsIt->second.size() > 1 && pointsIt->second[0]->getCurrentPosition() == pointsIt->second.back()->getCurrentPosition();
}



void SurfaceContoursWidget::highlightLastPoint( const std::shared_ptr<VisualObject>& obj )
{
    auto& contour = pickedPoints_[obj];
    int lastPointId = static_cast< int > ( contour.size() - 1 );
    if ( lastPointId > 0 )
    {
        updateBaseColor( contour[lastPointId - 1], params.ordinaryPointColor );
        if ( !isClosedCountour( obj ) )
            updateBaseColor( contour[lastPointId], params.lastPoitColor );
        else
            updateBaseColor( contour[lastPointId], params.closeContourPointColor );
    }
    else
        if ( lastPointId == 0 )
            updateBaseColor( contour[0], params.lastPoitColor ); // only one point in contour
}

std::pair<std::shared_ptr<MR::VisualObject>, int> SurfaceContoursWidget::getActivePoint() const
{
    return { activeObject_, activeIndex_ };
}

void SurfaceContoursWidget::setActivePoint( std::shared_ptr<MR::VisualObject> obj, int index )
{
    assert( pickedPoints_[obj].size() > index );

    // last point in closed contour are nonVisible and equal to first point. => 
    if ( isClosedCountour( obj ) && ( index >= pickedPoints_[obj].size() - 1 ) )
    {
        index = 0;
    }

    activeIndex_ = index;
    activeObject_ = obj;
}

std::shared_ptr<SurfacePointWidget> SurfaceContoursWidget::getActiveSurfacePoint() const
{
    if ( !activeObject_ )
        return {};
    assert( 0 <= activeIndex_ );

    const auto it = pickedPoints_.find( activeObject_ );
    assert( it != pickedPoints_.end() );
    const auto& contour = it->second;
    assert( activeIndex_ < contour.size() );
    return contour[activeIndex_];
}

bool SurfaceContoursWidget::appendPoint( const std::shared_ptr<VisualObject>& obj, const PickedPoint& triPoint )
{

    if ( !isObjectValidToPick_( obj ) )
        return false;

    auto onAddPointAction = [this, &obj, &triPoint] ()
    {
        auto& contour = pickedPoints_[obj];

        if ( params.writeHistory )
        {
            AppendHistory<AddPointActionPickerPoint>( *this, obj, triPoint );
        }

        contour.push_back( createPickWidget_( obj, triPoint ) );
        highlightLastPoint( obj );
        activeIndex_ = static_cast< int >( contour.size() - 1 );
        activeObject_ = obj;

        onPointAdd_( obj );
    };

    auto scopedBlock = getViewerInstance().getGlobalHistoryStore()->getScopeBlockPtr();
    if ( ( scopedBlock == nullptr ) && ( params.writeHistory ) )
    {
        SCOPED_HISTORY( "Add Point" + params.historySpecification );
        onAddPointAction();
    }
    else
        onAddPointAction();

    return true;
}

bool SurfaceContoursWidget::removePoint( const std::shared_ptr<VisualObject>& obj, int pickedIndex )
{

    auto onRemovePointAction = [this, &obj, pickedIndex] ()
    {
        auto& contour = pickedPoints_[obj];

        if ( params.writeHistory )
        {
            AppendHistory<RemovePointActionPickerPoint>( *this, obj, contour[pickedIndex]->getCurrentPosition(), pickedIndex );
        }
        contour.erase( contour.begin() + pickedIndex );
        activeIndex_ = pickedIndex;
        activeObject_ = obj;
        highlightLastPoint( obj );
        onPointRemove_( obj );
    };

    // for use add points and remove points in callback groups history actions
    auto scopedBlock = getViewerInstance().getGlobalHistoryStore()->getScopeBlockPtr();
    if ( ( scopedBlock == nullptr ) && ( params.writeHistory ) )
    {
        SCOPED_HISTORY( "Remove Point" + params.historySpecification );
        onRemovePointAction();
    }
    else
        onRemovePointAction();

    return true;
}

bool SurfaceContoursWidget::closeContour( const std::shared_ptr<VisualObject>& objectToCloseCoutour )
{
    if ( isClosedCountour( objectToCloseCoutour ) )
        return false;

    assert( objectToCloseCoutour != nullptr );
    auto triPoint = pickedPoints_[objectToCloseCoutour][0]->getCurrentPosition();
    appendPoint( objectToCloseCoutour, triPoint );
    activeIndex_ = 0;
    return true;
}



bool SurfaceContoursWidget::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( !isPickerActive_ )
        return false;

    if ( button != Viewer::MouseButton::Left )
        return false;

    auto allowExactPickFirst = params.surfacePointParams.pickInBackFaceObject;
    auto [obj, pick] = getViewerInstance().viewport().pick_render_object( allowExactPickFirst );

    if ( !obj )
        return false;

    if ( ( params.surfacePointParams.pickInBackFaceObject == false ) && ( SurfacePointWidget::isPickIntoBackFace( obj, pick, getViewerInstance().viewport().getCameraPoint() ) ) )
        return false;

    if ( !mod ) // just add new point 
    {
        auto objVisual = std::dynamic_pointer_cast< VisualObject >( obj );
        if ( !objVisual )
            return false;

        // all pick in point (without mod) must not comes here. Point widget must "eat" them. 
        if ( isClosedCountour( objVisual ) )
            return false;

        assert( objVisual != nullptr ); // contoursWidget_ can join for mesh objects only

        appendPoint( objVisual, pointOnObjectToPickedPoint( objVisual.get(), pick ) );
        return true;
    }
    else if ( mod == params.widgetContourCloseMod ) // close contour case 
    {
        // Try to find parent object 
        auto isFirstPointOnCountourClicked = false;
        std::shared_ptr<VisualObject> objectToCloseCoutour = nullptr;
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
        return closeContour( objectToCloseCoutour );
    }
    else if ( mod == params.widgetDeletePointMod )  // remove point case 
    {
        if ( pickedPoints_.empty() )
            return false;

        int pickedIndex = -1;
        std::shared_ptr<MR::VisualObject> pickedObj = nullptr;

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
                appendPoint( pickedObj, contour[0]->getCurrentPosition() );
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

    auto allowExactPickFirst = params.surfacePointParams.pickInBackFaceObject;
    auto [obj, pick] = getViewerInstance().viewport().pick_render_object( allowExactPickFirst );
    if ( !obj )
        return false;

    if ( ( params.surfacePointParams.pickInBackFaceObject == false ) && ( SurfacePointWidget::isPickIntoBackFace( obj, pick, getViewerInstance().viewport().getCameraPoint() ) ) )
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
    if ( params.writeHistory )
    {
        SCOPED_HISTORY( "Remove All Point" + params.historySpecification );
        for ( auto& [obj, contour] : pickedPoints_ )
            for ( int i = static_cast< int >( contour.size() - 1 ); i >= 0; --i )
                AppendHistory<RemovePointActionPickerPoint>( *this, obj, contour[i]->getCurrentPosition(), i );
    }
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