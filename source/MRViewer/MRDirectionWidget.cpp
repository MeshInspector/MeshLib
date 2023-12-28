#include "MRDirectionWidget.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRArrow.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRMatrix3Decompose.h"

namespace MR
{
    void DirectionWidget::create( const Vector3f& dir, const Vector3f& base, float length, OnDirectionChangedCallback onDirectionChanged, VisualObject* parent )
    {
        base_ = base;
        length_ = length;
        onDirectionChanged_ = onDirectionChanged;
        parent_ = parent;
        connect( &getViewerInstance(), 10, boost::signals2::at_front );
        updateDirection( dir );
    }

    void DirectionWidget::reset()
    {
        clear_();
        disconnect();
    }

    void DirectionWidget::updateDirection( const Vector3f& dir )
    {
        dir_ = dir.normalized();
        if ( !directionObj_ )
        {
            std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( {}, dir_ * length_, length_ * 0.02f, length_ * 0.04f, length_ * 0.08f ) );
            directionObj_ = std::make_shared<ObjectMesh>();
            directionObj_->setMesh( mesh );
            directionObj_->setAncillary( true );
            directionObj_->setFrontColor( color_, false );
            directionObj_->setFlatShading( true );

            if ( parent_ )
                parent_->addChild( directionObj_ );
            else
                SceneRoot::get().addChild( directionObj_ );
        }

        Matrix3f rot, scale;
        if ( parent_ )
            decomposeMatrix3( parent_->worldXf().A, rot, scale );
        directionObj_->setXf( AffineXf3f::translation( base_ ) * AffineXf3f::linear( rot.inverse() * Matrix3f::rotation( Vector3f::plusZ(), dir ) ) );
    }

    void DirectionWidget::updateArrow( const Vector3f& base, float length )
    {
        if ( !directionObj_ )
            return;

        base_ = base;
        length_ = length;
        std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( {}, dir_ * length_, length_ * 0.02f, length_ * 0.04f, length_ * 0.08f));
        directionObj_->setXf( AffineXf3f::translation( base_ ) * directionObj_->xf() );
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

        mousePressed_ = false;
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
        return true;
    }

    bool DirectionWidget::onMouseMove_( int x, int y )
    {
        if ( !mousePressed_ )
            return false;
        
        auto viewer = Viewer::instance();
        const auto viewportEnd = viewer->screenToViewport( Vector3f( float( x ), float( y ), 0.f ), viewer->viewport().id );
        const auto worldEndPoint = viewer->viewport().unprojectFromViewportSpace( { viewportEnd.x, viewportEnd.y, viewportStartPointZ_ } );
        const auto newDir = worldEndPoint - directionObj_->worldXf()( base_ );
        updateDirection( newDir );
        if ( onDirectionChanged_ )
            onDirectionChanged_( newDir, needToSaveHistory_ );
        
        needToSaveHistory_ = false;
        return true;
    }

    bool DirectionWidget::onMouseUp_( Viewer::MouseButton button, int )
    {
        if ( button != Viewer::MouseButton::Left )
            return false;

        if ( !mousePressed_ )
            return false;

        mousePressed_ = false;
        needToSaveHistory_ = true;
        return true;
    }

    void DirectionWidget::setColor( const Color& color )
    {
        color_ = color;
    }

    const Color& DirectionWidget::getColor() const
    {
        return color_;
    }
}
