#include "MRDirectionWidget.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRArrow.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeObjectAction.h"
#include "MRViewer/MRViewport.h"

namespace MR
{
    DirectionWidget::DirectionWidget( const Vector3f& dir, const Vector3f& base, float length, OnDirectionChangedCallback onDirectionChanged )
    : base_( base )
    , length_( length )
    , onDirectionChanged_( onDirectionChanged )
    {
        connect( &getViewerInstance(), 10, boost::signals2::at_front );
        updateDirection( dir, false );
    }

    DirectionWidget::~DirectionWidget()
    {
        FilterHistoryByCondition( [&] ( const std::shared_ptr<HistoryAction>& action ) -> bool
        {
            return bool( std::dynamic_pointer_cast<ChangeDirAction>( action ) );
        }, true );
        clear_();
        disconnect();
    }

    void DirectionWidget::updateDirection( const Vector3f& dir, bool needHistUpdate )
    {
        dir_ = dir.normalized();
        if ( !directionObj_ )
        {
            std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( {}, dir_ * length_, length_ * 0.02f, length_ * 0.04f, length_ * 0.08f ) );
            directionObj_ = std::make_shared<ObjectMesh>();
            directionObj_->setMesh( mesh );
            directionObj_->setAncillary( true );
            directionObj_->setFrontColor( Color::red(), false );
            directionObj_->setFlatShading( true );
            SceneRoot::get().addChild( directionObj_ );
        }
        if ( needHistUpdate )
            AppendHistory<ChangeDirAction>( *this, directionObj_ );

        auto transform = AffineXf3f::translation(base_) * AffineXf3f::linear(Matrix3f::rotation(Vector3f::plusZ(), dir));
        directionObj_->setXf( transform );
    }

    void DirectionWidget::updateArrow( const Vector3f& base, float length )
    {
        clear_();
        const auto oldBase = base_;
        base_ = base;
        length_ = length;
        std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( base_, base_ + dir_ * length_, length_ * 0.02f, length_ * 0.04f, length_ * 0.08f ) );
        directionObj_->setMesh( mesh );
        directionObj_->setXf( AffineXf3f::translation( base_ - oldBase ) * directionObj_->xf() );
    }

    void DirectionWidget::setVisible( bool visible )
    {
        directionObj_->setVisible( visible );
    }

    void DirectionWidget::clear_()
    {
        if ( directionObj_ )
        {
            directionObj_->detachFromParent();
            directionObj_.reset();
        }
    }

    bool DirectionWidget::onMouseDown_( Viewer::MouseButton button, int mod )
    {
        if ( button != Viewer::MouseButton::Left || mod != 0 )
            return false;

        auto viewer = Viewer::instance();
        viewer->select_hovered_viewport();
        const auto [obj, pof] = viewer->viewport().pick_render_object();
        if ( obj != directionObj_ )
            return false;

        mousePressed_ = true;
        worldStartPoint_ = directionObj_->worldXf()( pof.point );
        viewportStartPointZ_ = viewer->viewport().projectToViewportSpace( worldStartPoint_ ).z;

        AppendHistory<ChangeDirAction>( *this, directionObj_ );
        return true;
    }

    bool DirectionWidget::onMouseMove_( int x, int y )
    {
        if ( !mousePressed_ )
            return false;
        
        auto viewer = Viewer::instance();
        const auto viewportEnd = viewer->screenToViewport( Vector3f( float( x ), float( y ), 0.f ), viewer->viewport().id );
        const auto worldEndPoint = viewer->viewport().unprojectFromViewportSpace( { viewportEnd.x, viewportEnd.y, viewportStartPointZ_ } );
        const auto newDir = worldEndPoint - base_;
        updateDirection( newDir, false );
        if ( onDirectionChanged_ )
            onDirectionChanged_( newDir );

        return true;
    }

    bool DirectionWidget::onMouseUp_( Viewer::MouseButton button, int )
    {
        if ( button != Viewer::MouseButton::Left )
            return false;

        if ( !mousePressed_ )
            return false;

        mousePressed_ = false;
        return true;
    }

}
