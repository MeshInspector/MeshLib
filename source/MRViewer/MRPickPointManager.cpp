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

class PickPointManager::AddRemovePointHistoryAction : public PickPointManager::WidgetHistoryAction
{
public:
    /// appends new point at given position, and returns undo action for its removal
    static std::shared_ptr<AddRemovePointHistoryAction> appendAndGetUndo(
        PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point );

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

    void insertPoint_();
    void removePoint_();

    std::string name_;
    PickPointManager& widget_;
    const std::shared_ptr<MR::VisualObject> obj_;
    PickedPoint point_;
    int index_ = 0; // -1 here means insert after last one
    bool insertOnAction_ = false;
};

std::shared_ptr<PickPointManager::AddRemovePointHistoryAction> PickPointManager::AddRemovePointHistoryAction::appendAndGetUndo(
    PickPointManager& widget, const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& point )
{
    std::shared_ptr<AddRemovePointHistoryAction> res( new AddRemovePointHistoryAction( "Append Point", widget, obj, point, -1, true ) );
    res->insertPoint_();
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
        insertPoint_();
    else
        removePoint_();
}

void PickPointManager::AddRemovePointHistoryAction::insertPoint_()
{
    assert( insertOnAction_ );
    index_ = widget_.insertPointNoHistory_( obj_, index_, point_ );
    insertOnAction_ = false;
}

int PickPointManager::insertPointNoHistory_( const std::shared_ptr<VisualObject>& obj, int index, const PickedPoint& point )
{
    auto& contour = pickedPoints_[obj];
    if ( index < 0 )
        index = (int)contour.size();
    else
        assert( index <= contour.size() );

    contour.insert( contour.begin() + index, createPickWidget_( obj, point ) );
    if ( index + 1 == contour.size() ) // last point was added
        colorLast2Points_( obj );
    if ( params.onPointAdd )
        params.onPointAdd( obj, index );
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
    [[nodiscard]] virtual size_t heapBytes() const override;
private:
    PickPointManager& widget_;
    const std::shared_ptr<MR::VisualObject> obj_;
    PickedPoint point_;
    int index_;
};

std::string PickPointManager::MovePointHistoryAction::name() const
{
    return "Move point" + widget_.params.historyNameSuffix;
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

size_t PickPointManager::MovePointHistoryAction::heapBytes() const
{
    return 0; //this undo action will be deleted in widget disable
}

std::shared_ptr<SurfacePointWidget> PickPointManager::createPickWidget_( const std::shared_ptr<MR::VisualObject>& obj, const PickedPoint& pt )
{
    auto newPoint = std::make_shared<SurfacePointWidget>();
    newPoint->setAutoHover( false );
    newPoint->setParameters( params.surfacePointParams );
    newPoint->create( obj, pt );

    if ( connectionHolders_.find( obj ) == connectionHolders_.end() )
    {
        ConnectionHolder holder;
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

bool PickPointManager::isClosedCountour( const std::shared_ptr<VisualObject>& obj ) const
{
    auto pointsIt = pickedPoints_.find( obj );
    if ( pointsIt == pickedPoints_.end() )
        return false;
    auto & points = pointsIt->second;
    return points.size() > 1 && points[0]->getCurrentPosition() == points.back()->getCurrentPosition();
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
        if ( !isClosedCountour( obj ) )
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

bool PickPointManager::appendPoint( const std::shared_ptr<VisualObject>& obj, const PickedPoint& triPoint )
{
    auto onAddPointAction = [this, &obj, &triPoint] ()
    {
        auto actionPtr = AddRemovePointHistoryAction::appendAndGetUndo( *this, obj, triPoint );
        if ( params.writeHistory )
            AppendHistory( std::move( actionPtr ) );
    };

    if ( params.writeHistory )
    {
        SCOPED_HISTORY( "Append Point" + params.historyNameSuffix );
        onAddPointAction();
    }
    else
        onAddPointAction();

    return true;
}

bool PickPointManager::removePoint( const std::shared_ptr<VisualObject>& obj, int pickedIndex )
{
    auto onRemovePointAction = [this, &obj, pickedIndex] ()
    {
        auto actionPtr = AddRemovePointHistoryAction::removeAndGetUndo( *this, obj, pickedIndex );
        if ( params.writeHistory )
            AppendHistory( std::move( actionPtr ) );
    };

    // for use add points and remove points in callback groups history actions
    if ( params.writeHistory )
    {
        SCOPED_HISTORY( "Remove Point" + params.historyNameSuffix );
        onRemovePointAction();
    }
    else
        onRemovePointAction();

    return true;
}

bool PickPointManager::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( button != Viewer::MouseButton::Left )
        return false;

    auto [obj, pick] = pick_();
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
    if ( !pickObj )
        return false;

    if ( ( params.surfacePointParams.pickInBackFaceObject == false ) && ( SurfacePointWidget::isPickIntoBackFace( pickObj, pick, getViewerInstance().viewport().getCameraPoint() ) ) )
        return false;

    for ( const auto & [obj, widgets] : pickedPoints_ )
        for ( int index = 0; index < widgets.size(); ++index )
        {
            const auto& widget = widgets[index];
            bool hovered = pickObj == widget->getPickSphere();
            widget->setHovered( hovered );
            if ( hovered )
            {
                // setting callback is very cheap operation (in comparison to pick_ above),
                // and we do it here because here we know up-today index of the point
                widget->setStartMoveCallback( [this, obj = obj, index] ( SurfacePointWidget & pointWidget, const PickedPoint& point )
                {
                    const bool closedPath = isClosedCountour( obj );

                    if ( closedPath )
                    {
                        const auto& contour = pickedPoints_[obj];
                        if ( &pointWidget == contour[0].get() )
                        {
                            if ( params.writeHistory )
                            {
                                SCOPED_HISTORY( "Move point" + params.historyNameSuffix );
                                AppendHistory<MovePointHistoryAction>( *this, obj, point, index );
                                AppendHistory<MovePointHistoryAction>( *this, obj, point, int( contour.size() ) - 1 );
                            }
                            moveClosedPoint_ = true;
                        }
                        else
                        {
                            if ( params.writeHistory )
                                AppendHistory<MovePointHistoryAction>( *this, obj, point, index );
                        }
                    }
                    else
                    {
                        if ( params.writeHistory )
                            AppendHistory<MovePointHistoryAction>( *this, obj, point, index );
                    }
                    draggedPointWidget_ = &pointWidget;
                    if ( params.onPointMoveStart )
                        params.onPointMoveStart( obj, index );

                } );
                widget->setEndMoveCallback( [this, obj = obj, index] ( SurfacePointWidget & pointWidget, const PickedPoint& point )
                {
                    if ( moveClosedPoint_ )
                    {
                        const auto& contour = pickedPoints_[obj];
                        if ( &pointWidget == contour[0].get() )
                        {
                            contour.back()->setCurrentPosition( point );
                            moveClosedPoint_ = false;
                        }
                    }
                    assert( draggedPointWidget_ == &pointWidget );
                    draggedPointWidget_ = nullptr;
                    if ( params.onPointMoveFinish )
                        params.onPointMoveFinish( obj, index );
                } );
            }
        }
    return false;
}

PickPointManager::PickPointManager()
{
    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void PickPointManager::clear( bool writeHistory )
{
    if ( params.writeHistory && writeHistory )
        AppendHistory<ClearHistoryAction>( "Clear points" + params.historyNameSuffix, *this );

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
    draggedPointWidget_ = nullptr;
    connectionHolders_.clear();
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

class PickPointManager::ClearHistoryAction : public PickPointManager::WidgetHistoryAction
{
public:
    ClearHistoryAction( std::string name, PickPointManager& widget );

public:
    [[nodiscard]] std::string name() const override { return name_; }

    void action( Type type ) override;

    [[nodiscard]] size_t heapBytes() const override;

private:
    std::string name_;
    PickPointManager& widget_;

    struct ObjectState
    {
        std::weak_ptr<VisualObject> objPtr;
        std::vector<PickedPoint> pickedPoints;
    };
    std::vector<ObjectState> states_;
};

PickPointManager::ClearHistoryAction::ClearHistoryAction( std::string name, PickPointManager& widget )
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
}

void PickPointManager::ClearHistoryAction::action( Type type )
{
    MR_SCOPED_VALUE( widget_.params.writeHistory, false );

    switch ( type )
    {
        case Type::Undo:
            for ( const auto& state : states_ )
            {
                if ( const auto obj = state.objPtr.lock() )
                {
                    for ( const auto& p : state.pickedPoints )
                        widget_.insertPointNoHistory_( obj, -1, p );
                }
            }
            break;

        case Type::Redo:
            widget_.clear();
            break;
    }
}

size_t PickPointManager::ClearHistoryAction::heapBytes() const
{
    return 0; // this undo action will be deleted in widget disable
}

} // namespace MR
