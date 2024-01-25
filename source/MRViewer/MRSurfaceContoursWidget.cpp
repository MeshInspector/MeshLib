#include "MRSurfaceContoursWidget.h"
#include "MRViewport.h"
#include "MRMesh/MRHistoryAction.h"
#include "MRAppendHistory.h"
#include <GLFW/glfw3.h>
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
class AddPointActionPickerPoint : public HistoryAction
{
public:
    AddPointActionPickerPoint( SurfaceContoursWidget& plugin, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point ) :
        plugin_{ plugin },
        obj_{ obj },
        point_{ point }
    {};

    virtual std::string name() const override
    {
        return "Add Point";
    }
    virtual void action( Type actionType ) override
    {
        if ( !plugin_.isPickerActive_ )
            return;
        if ( actionType == Type::Undo )
        {
            plugin_.pickedPoints_[obj_].pop_back();

            if ( !plugin_.pickedPoints_[obj_].empty() )
                updateBaseColor( plugin_.pickedPoints_[obj_].back(), Color::green() );

            plugin_.activeIndex = int( plugin_.pickedPoints_[obj_].size() );
            plugin_.activeObject = obj_;

            plugin_.onPointRemove_( obj_ );
        }
        else
        {
            if ( !plugin_.pickedPoints_[obj_].empty() )
                updateBaseColor( plugin_.pickedPoints_[obj_].back(), Color::gray() );

            plugin_.pickedPoints_[obj_].push_back( plugin_.createPickWidget_( obj_, point_ ) );

            updateBaseColor( plugin_.pickedPoints_[obj_].back(), Color::green() );


            plugin_.onPointAdd_( obj_ );
        }

    }
    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return 0; //this undo action will be deleted in plugin disable
    }
private:
    SurfaceContoursWidget& plugin_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
};

class RemovePointActionPickerPoint : public HistoryAction
{
public:
    RemovePointActionPickerPoint( SurfaceContoursWidget& plugin, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point, int index ) :
        plugin_{ plugin },
        obj_{ obj },
        point_{ point },
        index_{ index }
    {};

    virtual std::string name() const override
    {
        return "Remove Point";
    }
    virtual void action( Type actionType ) override
    {
        if ( !plugin_.isPickerActive_ )
            return;

        if ( actionType == Type::Undo )
        {
            if ( index_ == plugin_.pickedPoints_[obj_].size() && !plugin_.pickedPoints_[obj_].empty() )
                updateBaseColor( plugin_.pickedPoints_[obj_].back(), Color::gray() );

            plugin_.pickedPoints_[obj_].insert( plugin_.pickedPoints_[obj_].begin() + index_, plugin_.createPickWidget_( obj_, point_ ) );

            if ( index_ + 1 == plugin_.pickedPoints_[obj_].size() )
                updateBaseColor( plugin_.pickedPoints_[obj_].back(), Color::green() );
            plugin_.activeIndex = index_;
            plugin_.activeObject = obj_;

            plugin_.pickedPoints_[obj_].back()->setHovered( false );
            plugin_.onPointAdd_( obj_ );
        }
        else
        {
            plugin_.pickedPoints_[obj_].erase( plugin_.pickedPoints_[obj_].begin() + index_ );

            if ( index_ == plugin_.pickedPoints_[obj_].size() && !plugin_.pickedPoints_[obj_].empty() )
                updateBaseColor( plugin_.pickedPoints_[obj_].back(), Color::green() );

            plugin_.activeIndex = index_;
            plugin_.activeObject = obj_;

            plugin_.onPointRemove_( obj_ );
        }
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return 0; //this undo action will be deleted in plugin disable
    }
private:
    SurfaceContoursWidget& plugin_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
    int index_;
};

class ChangePointActionPickerPoint : public HistoryAction
{
public:
    ChangePointActionPickerPoint( SurfaceContoursWidget& plugin, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point, int index ) :
        plugin_{ plugin },
        obj_{ obj },
        point_{ point },
        index_{ index }
    {};

    virtual std::string name() const override
    {
        return "Move Point";
    }
    virtual void action( Type ) override
    {
        if ( !plugin_.isPickerActive_ )
            return;

        plugin_.pickedPoints_[obj_][index_]->updateCurrentPosition( point_ );
        plugin_.activeIndex = index_;
        plugin_.activeObject = obj_;
        plugin_.onPointMoveFinish_( obj_ );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return 0; //this undo action will be deleted in plugin disable
    }
private:
    SurfaceContoursWidget& plugin_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
    int index_;
};

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

        if ( closedPath && curentPoint.lock() == pickedPoints_[obj][0] )
        {
            SCOPED_HISTORY( "Change Point" );
            AppendHistory<ChangePointActionPickerPoint>( *this, obj, point, activeIndex );
            AppendHistory<ChangePointActionPickerPoint>( *this, obj, point, int( pickedPoints_[obj].size() ) - 1 );
            moveClosedPoint_ = true;
        }
        else if ( !closedPath || curentPoint.lock() != pickedPoints_[obj].back() )
        {
            AppendHistory<ChangePointActionPickerPoint>( *this, obj, point, activeIndex );
        }
        activeChange_ = true;
        onPointMove_( obj );

    } );
    newPoint->setEndMoveCallback( [this, obj, curentPoint] ( const MeshTriPoint& point )
    {
        if ( moveClosedPoint_ && curentPoint.lock() == pickedPoints_[obj][0] )
        {
            pickedPoints_[obj].back()->updateCurrentPosition( point );
        }
        activeChange_ = false;
        onPointMoveFinish_( obj );
    } );

    return newPoint;
}

bool SurfaceContoursWidget::isClosedCountour( const std::shared_ptr<ObjectMeshHolder>& obj )
{
    return pickedPoints_[obj].size() > 1
        && pickedPoints_[obj][0]->getCurrentPosition() == pickedPoints_[obj].back()->getCurrentPosition();
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
        if ( !pickedPoints_[obj].empty() )
            updateBaseColor( pickedPoints_[obj].back(), Color::gray() );

        AppendHistory<AddPointActionPickerPoint>( *this, obj, triPoint );
        pickedPoints_[obj].push_back( createPickWidget_( obj, triPoint ) );
        updateBaseColor( pickedPoints_[obj].back(), close ? Color::transparent() : Color::green() );
        onPointAdd_( obj );
    };
    auto removePoint = [this] ( const std::shared_ptr<ObjectMeshHolder> obj, int pickedIndex )
    {
        if ( pickedIndex == int( pickedPoints_[obj].size() ) - 1 && pickedPoints_[obj].size() > 1 )
        {
            SurfacePointWidget::Parameters params;
            params.baseColor = Color::green();
            pickedPoints_[obj][pickedIndex - 1]->setParameters( params );
        }

        AppendHistory<RemovePointActionPickerPoint>( *this, obj, pickedPoints_[obj][pickedIndex]->getCurrentPosition(), pickedIndex );
        pickedPoints_[obj].erase( pickedPoints_[obj].begin() + pickedIndex );
        activeIndex = pickedIndex;
        activeObject = obj;
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
    else if ( mod == GLFW_MOD_CONTROL ) // close contour case 
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
        return true;
    }
    else if ( mod == GLFW_MOD_SHIFT )  // remove point case 
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
            assert( pickedIndex != pickedPoints_[pickedObj].size() - 1 ); // unable to pick point which is close countour

            SCOPED_HISTORY( "Remove Point" );

            // 4 points - minimal non-trivial closed path
            // last on is a "pseudo" point to close contour
            // so remove last point which is close contour in case of: 
            // 1) Contour can no longer been closed (only 2 valid point left ) 
            // 2) First point need to be remove. Last point will be restored later.  
            if ( pickedPoints_[pickedObj].size() == 4 || pickedIndex == 0 )
                removePoint( pickedObj, ( int )pickedPoints_[pickedObj].size() - 1 );

            // Remove point marked to be removed. 
            removePoint( pickedObj, pickedIndex );

            // Restore close countour. 
            if ( pickedPoints_[pickedObj].size() > 2 && pickedIndex == 0 )
                addPoint( pickedObj, pickedPoints_[pickedObj][0]->getCurrentPosition(), true );
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
                activeIndex = i;
                activeObject = contour.first;
            }
        }
    return false;
}

void SurfaceContoursWidget::create( PickerPointCallBack onPointAdd, PickerPointCallBack onPointMove, PickerPointCallBack onPointMoveFinish, PickerPointCallBack onPointRemove )
{
    onPointAdd_ = std::move( onPointAdd );
    onPointMove_ = std::move( onPointMove );
    onPointMoveFinish_ = std::move( onPointMoveFinish );
    onPointRemove_ = std::move( onPointRemove );

    clear();

    // 10 group to imitate plugins behavior
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void SurfaceContoursWidget::clear()
{
    pickedPoints_.clear();
    activeIndex = 0;
    activeObject = nullptr;
}

void SurfaceContoursWidget::reset()
{
    clear();
    enable( false );

    FilterHistoryByCondition( [&] ( const std::shared_ptr<HistoryAction>& action )
    {
        bool res = bool( std::dynamic_pointer_cast< AddPointActionPickerPoint >( action ) ) ||
            bool( std::dynamic_pointer_cast< RemovePointActionPickerPoint >( action ) ) ||
            bool( std::dynamic_pointer_cast< ChangePointActionPickerPoint >( action ) );
        return res;
    } );

    disconnect();
}


} // namespace MR 