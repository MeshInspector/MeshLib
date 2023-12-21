#include "MRDirectionWidget.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRArrow.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeObjectAction.h"

namespace MR
{
    DirectionWidget::DirectionWidget( const Vector3f& dir, const Vector3f& base, float length )
    : base_( base )
    , length_( length )
    {
        updateDirection( dir );
    }

    DirectionWidget::~DirectionWidget()
    {
        FilterHistoryByCondition( [&] ( const std::shared_ptr<HistoryAction>& action ) -> bool
        {
            return bool( std::dynamic_pointer_cast<ChangeDirAction>( action ) );
        }, true );
        clear_();
    }

    void DirectionWidget::updateDirection( const Vector3f& dir )
    {
        dir_ = dir;
        bool needHistUpdate = bool( directionObj_ );
        if ( !directionObj_ )
        {
            std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( base_, base_ + dir_ * length_, length_ * 0.02f, length_ * 0.04f, length_ * 0.08f ) );
            directionObj_ = std::make_shared<ObjectMesh>();
            directionObj_->setMesh( mesh );
            directionObj_->setAncillary( true );
            directionObj_->setFrontColor( Color::red(), false );
            directionObj_->setFlatShading( true );
            SceneRoot::get().addChild( directionObj_ );
        }
        if ( needHistUpdate )
            AppendHistory<ChangeDirAction>( *this, directionObj_ );

        auto transform = AffineXf3f::translation( base_ ) * AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), dir ) );
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

    void DirectionWidget::clear_()
    {
        if ( directionObj_ )
        {
            directionObj_->detachFromParent();
            directionObj_.reset();
        }
    }

}
