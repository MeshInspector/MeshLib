#include "MRSurfaceContoursWidget.h"
#include "MRMesh/MRScopedValue.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRHistoryAction.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRPointOnObject.h"
#include "MRMesh/MRHeapBytes.h"
#include "MRMesh/MRFinally.h"

namespace MR
{

void updateBaseColor( std::shared_ptr<SurfacePointWidget> point, const Color& color )
{
    auto params = point->getParameters();
    params.baseColor = color;
    point->setParameters( params );
}

// History classes;

std::string SurfaceContoursWidget::AddPointActionPickerPoint::name() const
{
    return "Pick point" + widget_.params.historyNameSuffix;
}

void SurfaceContoursWidget::AddPointActionPickerPoint::action( Type actionType )
{
    if ( !widget_.isPickerActive_ )
        return;

    MR_SCOPED_VALUE( widget_.undoRedoMode_, true );
    MR_SCOPED_VALUE( widget_.params.writeHistory, false );

    auto& contour = widget_.pickedPoints_[obj_];
    if ( actionType == Type::Undo )
    {
        widget_.surfacePointWidgetCache_.erase( contour.back()->getPickSphere().get() );
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
size_t SurfaceContoursWidget::AddPointActionPickerPoint::heapBytes() const
{
    return 0; //this undo action will be deleted in widget disable
}



std::string SurfaceContoursWidget::RemovePointActionPickerPoint::name() const
{
    return "Remove point" + widget_.params.historyNameSuffix;
}
void SurfaceContoursWidget::RemovePointActionPickerPoint::action( Type actionType )
{
    if ( !widget_.isPickerActive_ )
        return;

    MR_SCOPED_VALUE( widget_.undoRedoMode_, true );
    MR_SCOPED_VALUE( widget_.params.writeHistory, false );

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
        auto it = contour.begin() + index_;
        widget_.surfacePointWidgetCache_.erase( (*it)->getPickSphere().get() );
        contour.erase( it );

        widget_.activeIndex_ = index_;
        widget_.activeObject_ = obj_;
        widget_.highlightLastPoint( obj_ );
        widget_.onPointRemove_( obj_ );
    }
}

size_t SurfaceContoursWidget::RemovePointActionPickerPoint::heapBytes() const
{
    return 0; //this undo action will be deleted in widget disable
}

std::string SurfaceContoursWidget::ChangePointActionPickerPoint::name() const
{
    return "Move point" + widget_.params.historyNameSuffix;
}

void SurfaceContoursWidget::ChangePointActionPickerPoint::action( Type )
{
    if ( !widget_.isPickerActive_ )
        return;

    MR_SCOPED_VALUE( widget_.undoRedoMode_, true );
    MR_SCOPED_VALUE( widget_.params.writeHistory, false );

    widget_.pickedPoints_[obj_][index_]->updateCurrentPosition( point_ );
    widget_.activeIndex_ = index_;
    widget_.activeObject_ = obj_;
    widget_.onPointMoveFinish_( obj_ );
}

size_t SurfaceContoursWidget::ChangePointActionPickerPoint::heapBytes() const
{
    return 0; //this undo action will be deleted in widget disable
}

void SurfaceContoursWidget::enable( bool isEnabled )
{
    isPickerActive_ = isEnabled;
    if ( !isPickerActive_ )
        clear( false );
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
                    SCOPED_HISTORY( "Move point" + params.historyNameSuffix );
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

    if ( surfaceConnectionHolders_.find( obj ) == surfaceConnectionHolders_.end() )
    {
        SurfaceConnectionHolder holder;
        // re-validate the picked points on object's change
        auto updatePoints = [this, objPtr = std::weak_ptr( obj )] ( std::uint32_t )
        {
            if ( auto obj = objPtr.lock() )
            {
                auto& points = pickedPoints_[obj];
                const auto pointCount = points.size();
                for ( auto i = (int)pointCount - 1; i >= 0; --i )
                {
                    auto& point = points[i];
                    const auto& pos = point->getCurrentPosition();
                    if ( isPickedPointValid( obj.get(), pos ) )
                        point->updateCurrentPosition( pos );
                    else
                        removePoint( obj, i );
                }
            }
        };
        if ( const auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( obj ) )
            holder.onMeshChanged = objMesh->meshChangedSignal.connect( updatePoints );
        else if ( const auto objPoints = std::dynamic_pointer_cast<ObjectPoints>( obj ) )
            holder.onPointsChanged = objPoints->pointsChangedSignal.connect( updatePoints );
        surfaceConnectionHolders_.emplace( obj, std::move( holder ) );
    }

    surfacePointWidgetCache_.emplace( newPoint->getPickSphere().get() );

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
    if ( !undoRedoMode_ && !isObjectValidToPick_( obj ) )
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
        SCOPED_HISTORY( "Pick point" + params.historyNameSuffix );
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
        surfacePointWidgetCache_.erase( contour[pickedIndex]->getPickSphere().get() );
        contour.erase( contour.begin() + pickedIndex );
        assert( contour.size() > pickedIndex );
        activeIndex_ = pickedIndex;
        activeObject_ = obj;
        highlightLastPoint( obj );
        onPointRemove_( obj );
    };

    // for use add points and remove points in callback groups history actions
    auto scopedBlock = getViewerInstance().getGlobalHistoryStore()->getScopeBlockPtr();
    if ( ( scopedBlock == nullptr ) && ( params.writeHistory ) )
    {
        SCOPED_HISTORY( "Remove point" + params.historyNameSuffix );
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

    Viewport::PickRenderObjectPredicate predicate;
    if ( params.pickPredicate )
    {
        predicate = [&] ( const VisualObject* visObj, ViewportMask mask )
        {
            // always keep the picked points pickable
            if ( surfacePointWidgetCache_.find( visObj ) != surfacePointWidgetCache_.end() )
                return true;
            return params.pickPredicate( visObj, mask );
        };
    }
    auto [obj, pick] = getViewerInstance().viewport().pickRenderObject( {
        .predicate = predicate,
        .exactPickFirst = params.surfacePointParams.pickInBackFaceObject,
    } );
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

        return appendPoint( objVisual, pointOnObjectToPickedPoint( objVisual.get(), pick ) );
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
            assert( pickedIndex != contour.size() - 1 ); // unable to pick point which is close contour

            std::unique_ptr<ScopeHistory> historyGuiard;
            if ( params.writeHistory )
                historyGuiard = std::make_unique<ScopeHistory>( "Remove point" + params.historyNameSuffix );

            // 4 points - minimal non-trivial closed path
            // last on is a "pseudo" point to close contour
            // so remove last point which is close contour in case of:
            // 1) Contour can no longer been closed (only 2 valid point left )
            // 2) First point need to be remove. Last point will be restored later.
            if ( contour.size() == 4 || pickedIndex == 0 )
                removePoint( pickedObj, ( int )contour.size() - 1 );

            // Remove point marked to be removed.
            removePoint( pickedObj, pickedIndex );

            // Restore close contour.
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

    Viewport::PickRenderObjectPredicate predicate;
    if ( params.pickPredicate )
    {
        predicate = [&] ( const VisualObject* visObj, ViewportMask mask )
        {
            // always keep the picked points pickable
            if ( surfacePointWidgetCache_.find( visObj ) != surfacePointWidgetCache_.end() )
                return true;
            return params.pickPredicate( visObj, mask );
        };
    }
    auto [obj, pick] = getViewerInstance().viewport().pickRenderObject( {
        .predicate = predicate,
        .exactPickFirst = params.surfacePointParams.pickInBackFaceObject,
    } );
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

    clear( false );

    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void SurfaceContoursWidget::clear( bool writeHistory )
{
    if ( params.writeHistory && writeHistory )
        AppendHistory<SurfaceContoursWidgetClearAction>( "Clear points" + params.historyNameSuffix, *this );

    while ( !pickedPoints_.empty() )
    {
        auto obj = pickedPoints_.begin()->first;
        pickedPoints_.erase( pickedPoints_.begin() );
        onPointRemove_( obj );
    }
    surfacePointWidgetCache_.clear();
    surfaceConnectionHolders_.clear();
    activeIndex_ = 0;
    activeObject_ = nullptr;
}

void SurfaceContoursWidget::reset()
{
    clear( false );
    enable( false );

    if ( ( params.writeHistory ) && ( params.filterHistoryonReset ) )
    {
        FilterHistoryByCondition( [&] ( const std::shared_ptr<HistoryAction>& action )
        {
            return bool( dynamic_cast<const WidgetHistoryAction *>( action.get() ) );
        } );
    }

    disconnect();
}

SurfaceContoursWidget::SurfaceContoursWidgetClearAction::SurfaceContoursWidgetClearAction( std::string name, SurfaceContoursWidget& widget )
    : name_( std::move( name ) )
    , widget_( widget )
{
    for ( const auto& [obj, contour] : widget_.pickedPoints_ )
    {
        ObjectState state;
        state.objPtr = obj;
        state.pickedPoints.reserve( contour.size() );
        for ( const auto& p : contour )
            state.pickedPoints.emplace_back( p->getCurrentPosition() );
        states_.emplace_back( std::move( state ) );
    }

    if ( widget_.activeObject_ )
    {
        assert( widget_.pickedPoints_.contains( widget_.activeObject_ ) );
        assert( widget_.activeIndex_ < widget_.pickedPoints_.at( widget_.activeObject_ ).size() );
        activeObject_ = widget_.activeObject_;
        activeIndex_ = widget_.activeIndex_;
    }
}

void SurfaceContoursWidget::SurfaceContoursWidgetClearAction::action( Type type )
{
    if ( !widget_.isPickerActive_ )
        return;

    MR_SCOPED_VALUE( widget_.undoRedoMode_, true );
    MR_SCOPED_VALUE( widget_.params.writeHistory, false );

    switch ( type )
    {
        case Type::Undo:
            for ( const auto& state : states_ )
            {
                if ( const auto obj = state.objPtr.lock() )
                {
                    for ( const auto& p : state.pickedPoints )
                        widget_.appendPoint( obj, p );
                }
            }
            if ( const auto activeObject = activeObject_.lock() )
            {
                widget_.setActivePoint( activeObject, activeIndex_ );
            }
            break;

        case Type::Redo:
            widget_.clear();
            break;
    }
}

size_t SurfaceContoursWidget::SurfaceContoursWidgetClearAction::heapBytes() const
{
    return 0; // this undo action will be deleted in widget disable
}

} // namespace MR
