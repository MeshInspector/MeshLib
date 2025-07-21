#include "MRPickPointManager.h"
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
#include "MRMesh/MRScopedValue.h"

namespace MR
{

class PickPointManager::SetStateHistoryAction : public PickPointManager::WidgetHistoryAction
{
public:
    /// clears given PickPointManager, and returns undo action for restoring original state
    static std::shared_ptr<SetStateHistoryAction> clearAndGetUndo( PickPointManager& widget );

    /// sets given state to given PickPointManager, and returns undo action for restoring original state
    static std::shared_ptr<SetStateHistoryAction> setAndGetUndo( PickPointManager& widget, FullState&& fullState );

    virtual std::string name() const override { return name_ + widget_.params.historyNameSuffix; }
    void action( Type type ) override;
    [[nodiscard]] virtual size_t heapBytes() const override { return 0; } //this undo action will be deleted in widget disable

private:
    SetStateHistoryAction( std::string name, PickPointManager& widget, FullState&& fullState );

private:
    std::string name_;
    PickPointManager& widget_;
    FullState fullState_;
};

std::shared_ptr<PickPointManager::SetStateHistoryAction> PickPointManager::SetStateHistoryAction::clearAndGetUndo( PickPointManager& widget )
{
    std::shared_ptr<SetStateHistoryAction> res( new SetStateHistoryAction( "Clear Pick Points", widget, widget.getFullState() ) );
    widget.clearNoHistory_();
    return res;
}

std::shared_ptr<PickPointManager::SetStateHistoryAction> PickPointManager::SetStateHistoryAction::setAndGetUndo( PickPointManager& widget, FullState&& fullState )
{
    std::shared_ptr<SetStateHistoryAction> res( new SetStateHistoryAction( "Set Pick Points", widget, std::move( fullState ) ) );
    res->action( HistoryAction::Type::Redo ); // actually perform state setting
    return res;
}

PickPointManager::SetStateHistoryAction::SetStateHistoryAction( std::string name, PickPointManager& widget, FullState&& fullState )
    : name_( std::move( name ) )
    , widget_( widget )
    , fullState_( std::move( fullState ) )
{
}

void PickPointManager::SetStateHistoryAction::action( Type )
{
    widget_.swapStateNoHistory_( fullState_ );
}

class PickPointManager::AddRemovePointHistoryAction : public PickPointManager::WidgetHistoryAction
{
public:
    /// appends new point at the end, and returns undo action for its removal
    static std::shared_ptr<AddRemovePointHistoryAction> appendAndGetUndo(
        PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point, bool startDragging );

    /// inserts new point before given position, and returns undo action for its removal
    static std::shared_ptr<AddRemovePointHistoryAction> insertAndGetUndo(
        PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, int index, const PickedPoint& point, bool startDragging );

    /// removes point by index, and returns undo action for its addition at the same place
    static std::shared_ptr<AddRemovePointHistoryAction> removeAndGetUndo(
        PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, int index );

    virtual std::string name() const override { return name_ + widget_.params.historyNameSuffix; }
    virtual void action( Type actionType ) override;
    [[nodiscard]] virtual size_t heapBytes() const override { return 0; } //this undo action will be deleted in widget disable

private:
    AddRemovePointHistoryAction( std::string name, PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point, int index, bool insertOnAction ) :
        name_{ std::move( name ) },
        widget_{ widget },
        obj_{ obj },
        point_{ point },
        index_{ index },
        insertOnAction_{ insertOnAction }
    {};

    void insertPoint_( bool startDragging );
    void removePoint_();

    std::string name_;
    PickPointManager& widget_;
    const std::shared_ptr<MR::VisualObject> obj_;
    PickedPoint point_;
    int index_ = 0; // -1 here means insert after last one
    bool insertOnAction_ = false;
};

std::shared_ptr<PickPointManager::AddRemovePointHistoryAction> PickPointManager::AddRemovePointHistoryAction::appendAndGetUndo(
    PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point, bool startDragging )
{
    std::shared_ptr<AddRemovePointHistoryAction> res( new AddRemovePointHistoryAction( "Append Point", widget, obj, point, -1, true ) );
    res->insertPoint_( startDragging );
    return res;
}

std::shared_ptr<PickPointManager::AddRemovePointHistoryAction> PickPointManager::AddRemovePointHistoryAction::insertAndGetUndo(
    PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, int index, const PickedPoint& point, bool startDragging )
{
    std::shared_ptr<AddRemovePointHistoryAction> res( new AddRemovePointHistoryAction( "Insert Point", widget, obj, point, index, true ) );
    res->insertPoint_( startDragging );
    return res;
}

std::shared_ptr<PickPointManager::AddRemovePointHistoryAction> PickPointManager::AddRemovePointHistoryAction::removeAndGetUndo(
    PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, int index )
{
    assert( index >= 0 );
    std::shared_ptr<AddRemovePointHistoryAction> res( new AddRemovePointHistoryAction( "Remove Point", widget, obj, PickedPoint{}, index, false ) );
    res->removePoint_();
    return res;
}

void PickPointManager::AddRemovePointHistoryAction::action( Type )
{
    if ( insertOnAction_ )
        insertPoint_( false );
    else
        removePoint_();
}

void PickPointManager::AddRemovePointHistoryAction::insertPoint_( bool startDragging )
{
    assert( insertOnAction_ );
    index_ = widget_.insertPointNoHistory_( obj_, index_, point_, startDragging );
    insertOnAction_ = false;
}

int PickPointManager::insertPointNoHistory_( const std::shared_ptr<VisualObject>& obj, int index, const PickedPoint& point, bool startDragging )
{
    auto& contour = pickedPoints_[obj];
    if ( index < 0 )
        index = (int)contour.size();
    else
        assert( index <= contour.size() );

    auto pw = createPickWidget_( obj, point );
    contour.insert( contour.begin() + index, pw );
    if ( index + 1 == contour.size() ) // last point was added
        colorLast2Points_( obj );
    if ( params.onPointAdd )
        params.onPointAdd( obj, index );
    setHoveredPointWidget_( pw.get() );
    if ( startDragging )
    {
        MR_SCOPED_VALUE( params.writeHistory, false );
        pw->startDragging();
    }
    return index;
}

void PickPointManager::AddRemovePointHistoryAction::removePoint_()
{
    assert( !insertOnAction_ );
    point_ = widget_.removePointNoHistory_( obj_, index_ );
    insertOnAction_ = true;
}

PickedPoint PickPointManager::removePointNoHistory_( const std::shared_ptr<VisualObject>& obj, int index )
{
    auto& contour = pickedPoints_[obj];
    assert( index >= 0 );
    auto it = contour.begin() + index;
    auto point = (*it)->getCurrentPosition();
    const VisualObject * pickSphere = (*it)->getPickSphere().get();
    myPickSpheres_.erase( pickSphere );
    if ( draggedPointWidget_ == it->get() )
        draggedPointWidget_ = nullptr;
    if ( hoveredPointWidget_ == it->get() )
        hoveredPointWidget_ = nullptr;
    contour.erase( it );

    if ( index  == contour.size() ) // last point was deleted
        colorLast2Points_( obj );
    if ( params.onPointRemove )
        params.onPointRemove( obj, index );
    return point;
}

class PickPointManager::MovePointHistoryAction : public PickPointManager::WidgetHistoryAction
{
public:
    MovePointHistoryAction( PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point, int index ) :
        widget_{ widget },
        obj_{ obj },
        point_{ point },
        index_{ index }
    {};

    virtual std::string name() const override;
    virtual void action( Type ) override;
    [[nodiscard]] virtual size_t heapBytes() const override { return 0; } //this undo action will be deleted in widget disable

private:
    PickPointManager& widget_;
    const std::shared_ptr<MR::VisualObject> obj_;
    PickedPoint point_;
    int index_;
};

std::string PickPointManager::MovePointHistoryAction::name() const
{
    return "Move Point" + widget_.params.historyNameSuffix;
}

void PickPointManager::MovePointHistoryAction::action( Type )
{
    if ( auto w = widget_.getPointWidget( obj_, index_ ) )
    {
        w->swapCurrentPosition( point_ );
        if ( widget_.params.onPointMoveFinish )
            widget_.params.onPointMoveFinish( obj_, index_ );
    }
    else
        assert( false );
}

std::shared_ptr<SurfacePointWidget> PickPointManager::createPickWidget_( const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& pt )
{
    auto newPoint = std::make_shared<SurfacePointWidget>();
    newPoint->setAutoHover( false );
    newPoint->setParameters( params.surfacePointParams );
    newPoint->setBaseColor( params.ordinaryPointColor );
    newPoint->create( obj, pt );

    newPoint->setStartMoveCallback( [this, obj = obj] ( SurfacePointWidget & pointWidget, const PickedPoint& point )
    {
        const int index = getPointIndex( obj, pointWidget );
        if ( index < 0 )
        {
            assert( false );
            return;
        }
        moveClosedPoint_ = false;
        const auto& contour = pickedPoints_[obj];
        if ( isClosedContour( obj ) )
            moveClosedPoint_ = &pointWidget == contour[0].get();
        if ( params.writeHistory )
        {
            SCOPED_HISTORY( "Move Point" + params.historyNameSuffix );
            AppendHistory<MovePointHistoryAction>( *this, obj, point, index );
            if ( moveClosedPoint_ )
                AppendHistory<MovePointHistoryAction>( *this, obj, point, int( contour.size() ) - 1 );
        }
        assert( hoveredPointWidget_ == &pointWidget );
        draggedPointWidget_ = &pointWidget;
        if ( params.onPointMoveStart )
            params.onPointMoveStart( obj, index );

    } );

    newPoint->setOnMoveCallback( [this, obj = obj] ( SurfacePointWidget & pointWidget, const PickedPoint& point )
    {
        if ( moveClosedPoint_ )
        {
            const auto& contour = pickedPoints_[obj];
            if ( &pointWidget == contour[0].get() )
                contour.back()->setCurrentPosition( point );
        }
        assert( draggedPointWidget_ == &pointWidget );
        if ( params.onPointMove )
        {
            const int index = getPointIndex( obj, pointWidget );
            if ( index >= 0 )
                params.onPointMove( obj, index );
            else
                assert( false );
        }
    } );

    newPoint->setEndMoveCallback( [this, obj = obj] ( [[maybe_unused]] SurfacePointWidget & pointWidget, const PickedPoint& )
    {
        assert( draggedPointWidget_ == &pointWidget );
        draggedPointWidget_ = nullptr;
        if ( params.onPointMoveFinish )
        {
            const int index = getPointIndex( obj, pointWidget );
            if ( index >= 0 )
                params.onPointMoveFinish( obj, index );
            else
                assert( false );
        }
    } );

    if ( connectionHolders_.find( obj ) == connectionHolders_.end() )
    {
        ConnectionHolder holder;
        // re-validate the picked points on object's change
        auto updatePoints = [this, objPtr = std::weak_ptr( obj )] ( std::uint32_t )
        {
            if ( auto obj = objPtr.lock() )
            {
                if ( params.onUpdatePoints && !params.onUpdatePoints( obj ) )
                    return;
                auto& points = pickedPoints_[obj];
                const auto pointCount = points.size();
                for ( auto i = (int)pointCount - 1; i >= 0; --i )
                {
                    auto& point = points[i];
                    const auto& pos = point->getCurrentPosition();
                    if ( getPickedPointPosition( *obj, pos ).has_value() )
                        point->setCurrentPosition( pos ); // updates coordinates even for the same pos
                    else
                        removePoint( obj, i ); // questionable, but how to get here?
                }
            }
        };
        if ( const auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( obj ) )
            holder.onMeshChanged = objMesh->meshChangedSignal.connect( updatePoints );
        else if ( const auto objPoints = std::dynamic_pointer_cast<ObjectPoints>( obj ) )
            holder.onPointsChanged = objPoints->pointsChangedSignal.connect( updatePoints );
        connectionHolders_.emplace( obj, std::move( holder ) );
    }

    myPickSpheres_.emplace( newPoint->getPickSphere().get() );

    return newPoint;
}

bool PickPointManager::isClosedContour( const std::shared_ptr<VisualObject>& obj ) const
{
    auto pointsIt = pickedPoints_.find( obj );
    if ( pointsIt == pickedPoints_.end() )
        return false;
    auto & points = pointsIt->second;
    return points.size() > 1 && points[0]->getCurrentPosition() == points.back()->getCurrentPosition();
}

size_t PickPointManager::numPickPoints( const std::shared_ptr<VisualObject>& obj ) const
{
    auto pointsIt = pickedPoints_.find( obj );
    if ( pointsIt == pickedPoints_.end() )
        return 0;
    return pointsIt->second.size();
}

bool PickPointManager::closeContour( const std::shared_ptr<VisualObject>& obj, bool makeClosed )
{
    auto pointsIt = pickedPoints_.find( obj );
    if ( pointsIt == pickedPoints_.end() )
        return false;
    auto & points = pointsIt->second;
    if ( points.size() <= 1 )
        return false; // not enough to close or open

    if ( makeClosed )
    {
        if ( points[0]->getCurrentPosition() == points.back()->getCurrentPosition() )
            return false; // already closed

        auto triPoint = points[0]->getCurrentPosition();
        appendPoint( obj, triPoint );
        return true;
    }

    // make open
    if ( points[0]->getCurrentPosition() != points.back()->getCurrentPosition() )
        return false; // already open

    removePoint( obj, int( points.size() - 1 ) );
    return true;
}

void PickPointManager::colorLast2Points_( const std::shared_ptr<VisualObject>& obj )
{
    auto& contour = pickedPoints_[obj];
    int lastPointId = static_cast< int > ( contour.size() - 1 );
    if ( lastPointId > 0 )
    {
        contour[lastPointId - 1]->setBaseColor( params.ordinaryPointColor );
        if ( !isClosedContour( obj ) )
            contour[lastPointId]->setBaseColor( params.lastPointColor );
        else
            contour[lastPointId]->setBaseColor( params.closeContourPointColor );
    }
    else if ( lastPointId == 0 )
        contour[0]->setBaseColor( params.lastPointColor ); // only one point in contour
}

std::shared_ptr<SurfacePointWidget> PickPointManager::getPointWidget( const std::shared_ptr<VisualObject>& obj, int index ) const
{
    assert( obj );
    assert( index >= 0 );
    std::shared_ptr<SurfacePointWidget> res;
    const auto it = pickedPoints_.find( obj );
    if ( it != pickedPoints_.end() )
    {
        const auto& contour = it->second;
        assert( index < contour.size() );
        if ( index < contour.size() )
            res = contour[index];
    }
    return res;
}

int PickPointManager::getPointIndex( const std::shared_ptr<VisualObject>& obj, SurfacePointWidget& pointWidget ) const
{
    assert( obj );
    int res = 01;
    const auto it = pickedPoints_.find( obj );
    if ( it != pickedPoints_.end() )
    {
        const auto& contour = it->second;
        for ( int i = 0; i < contour.size(); ++i )
        {
            if ( contour[i].get() == &pointWidget )
            {
                res = i;
                break;
            }
        }
    }
    return res;
}

bool PickPointManager::appendPoint( const std::shared_ptr<VisualObject>& obj, const PickedPoint& triPoint, bool startDragging )
{
    appendHistory_( AddRemovePointHistoryAction::appendAndGetUndo( *this, obj, triPoint, startDragging ) );
    return true;
}

bool PickPointManager::insertPoint( const std::shared_ptr<VisualObject>& obj, int index, const PickedPoint& triPoint, bool startDragging )
{
    appendHistory_( AddRemovePointHistoryAction::insertAndGetUndo( *this, obj, index, triPoint, startDragging ) );
    return true;
}

bool PickPointManager::removePoint( const std::shared_ptr<VisualObject>& obj, int pickedIndex )
{
    appendHistory_( AddRemovePointHistoryAction::removeAndGetUndo( *this, obj, pickedIndex ) );
    return true;
}

bool PickPointManager::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( button != Viewer::MouseButton::Left )
        return false;

    auto [obj, pick] = pick_();
    if ( !obj )
        return false;

    if ( mod == 0 && myPickSpheres_.contains( obj.get() ) )
        return false; // we can be here only if pick sphere under cursor was not properly hovered and thus ignored mouse down

    if ( ( params.surfacePointParams.pickInBackFaceObject == false ) && ( SurfacePointWidget::isPickIntoBackFace( obj, pick, getViewerInstance().viewport().getCameraPoint() ) ) )
        return false;

    if ( !mod ) // just add new point
    {
        auto objVisual = std::dynamic_pointer_cast< VisualObject >( obj );
        if ( !objVisual )
            return false;

        // all pick in point (without mod) must not comes here. Point widget must "eat" them.
        if ( isClosedContour( objVisual ) )
            return false;

        assert( objVisual != nullptr ); // contoursWidget_ can join for mesh objects only

        if ( params.canAddPoint && !params.canAddPoint( objVisual, -1 ) )
            return false;
        return appendPoint( objVisual, pointOnObjectToPickedPoint( objVisual.get(), pick ), params.startDraggingJustAddedPoint );
    }
    else if ( mod == params.widgetContourCloseMod ) // close contour case
    {
        // Try to find parent object
        auto isFirstPointOnContourClicked = false;
        std::shared_ptr<VisualObject> objectToCloseCoutour = nullptr;
        for ( const auto& [parentObj, contour] : pickedPoints_ )
        {
            if ( contour.size() > 2 && obj == contour[0]->getPickSphere() )
            {
                isFirstPointOnContourClicked = true;
                objectToCloseCoutour = parentObj;
                break;
            }
        }
        if ( !isFirstPointOnContourClicked )
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

        if ( params.canRemovePoint && !params.canRemovePoint( obj, pickedIndex ) )
            return false;

        if ( isClosedContour( pickedObj ) )
        {
            assert( pickedIndex >= 0 );
            assert( pickedObj != nullptr );

            auto& contour = pickedPoints_[pickedObj];
            assert( pickedIndex != contour.size() - 1 ); // unable to pick point which is close contour

            std::unique_ptr<ScopeHistory> historyGuiard;
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

ObjAndPick PickPointManager::pick_() const
{
    Viewport::PickRenderObjectPredicate predicate;
    if ( params.pickPredicate )
    {
        predicate = [&] ( const VisualObject* visObj, ViewportMask mask )
        {
            // always keep my spheres pickable
            if ( myPickSpheres_.find( visObj ) != myPickSpheres_.end() )
                return true;
            return params.pickPredicate( visObj, mask );
        };
    }
    return getViewerInstance().viewport().pickRenderObject( {
        .predicate = predicate,
        .exactPickFirst = params.surfacePointParams.pickInBackFaceObject,
    } );
}

bool PickPointManager::onMouseMove_( int, int )
{
    if ( pickedPoints_.empty() || draggedPointWidget_ )
        return false;

    auto [pickObj, pick] = pick_();
    if ( hoveredPointWidget_ && pickObj != hoveredPointWidget_->getPickSphere() )
        setHoveredPointWidget_( nullptr );
    if ( !pickObj )
        return false;

    if ( ( params.surfacePointParams.pickInBackFaceObject == false ) && ( SurfacePointWidget::isPickIntoBackFace( pickObj, pick, getViewerInstance().viewport().getCameraPoint() ) ) )
        return false;

    for ( const auto & [obj, widgets] : pickedPoints_ )
    {
        if ( hoveredPointWidget_ )
            break;
        for ( int index = 0; index < widgets.size(); ++index )
        {
            const auto& widget = widgets[index];
            if ( pickObj == widget->getPickSphere() )
            {
                setHoveredPointWidget_( widget.get() );
                break;
            }
        }
    }
    return false;
}

PickPointManager::PickPointManager()
{
    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

auto PickPointManager::getFullState() const -> FullState
{
    std::vector<ObjectState> res;
    for ( const auto& [obj, contour] : pickedPoints_ )
    {
        ObjectState state;
        state.objPtr = obj;
        state.pickedPoints.reserve( contour.size() );
        for ( const auto& p : contour )
            state.pickedPoints.emplace_back( p->getCurrentPosition() );
        res.emplace_back( std::move( state ) );
    }
    return res;
}

void PickPointManager::clearNoHistory_()
{
    for ( auto& [obj, contour] : pickedPoints_ )
    {
        for ( int pickedIndex = int( contour.size() ) - 1; pickedIndex >= 0; --pickedIndex )
        {
            myPickSpheres_.erase( contour[pickedIndex]->getPickSphere().get() );
            contour.erase( contour.begin() + pickedIndex );
            assert( contour.size() >= pickedIndex );
            if ( params.onPointRemove )
                params.onPointRemove( obj, pickedIndex );
        }
    }
    pickedPoints_.clear();
    myPickSpheres_.clear();
    hoveredPointWidget_ = nullptr;
    draggedPointWidget_ = nullptr;
    connectionHolders_.clear();
}

void PickPointManager::swapStateNoHistory_( FullState& s )
{
    auto orgiginal = getFullState();

    clearNoHistory_();
    for ( const auto& state : s )
    {
        if ( const auto obj = state.objPtr.lock() )
        {
            for ( const auto& p : state.pickedPoints )
                insertPointNoHistory_( obj, -1, p, false );
        }
    }
    s = std::move( orgiginal );
}

void PickPointManager::clear()
{
    appendHistory_( SetStateHistoryAction::clearAndGetUndo( *this ) );
}

void PickPointManager::setFullState( FullState s )
{
    appendHistory_( SetStateHistoryAction::setAndGetUndo( *this, std::move( s ) ) );
}

PickPointManager::~PickPointManager()
{
    if ( params.writeHistory )
    {
        FilterHistoryByCondition( [&] ( const std::shared_ptr<HistoryAction>& action )
        {
            return bool( dynamic_cast<const WidgetHistoryAction *>( action.get() ) );
        } );
    }
    disconnect();
}

template<class HistoryActionType, typename... Args>
void PickPointManager::appendHistory_( Args&&... args )
{
    if ( params.writeHistory )
        AppendHistory<HistoryActionType>( std::forward<Args>( args )... );
}

void PickPointManager::appendHistory_( std::shared_ptr<HistoryAction> action ) const
{
    if ( params.writeHistory )
        AppendHistory( std::move( action ) );
}

void PickPointManager::setHoveredPointWidget_( SurfacePointWidget* newHoveredPoint )
{
    if ( hoveredPointWidget_ == newHoveredPoint )
        return;
    if ( hoveredPointWidget_ )
        hoveredPointWidget_->setHovered( false );
    hoveredPointWidget_ = newHoveredPoint;
    if ( hoveredPointWidget_ )
        hoveredPointWidget_->setHovered( true );
}

} // namespace MR
