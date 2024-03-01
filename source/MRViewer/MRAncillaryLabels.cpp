#include "MRAncillaryLabels.h"
#include "MRMesh/MRObjectLabel.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

void AncillaryLabel::make( Object& parent, const PositionedText& text, bool depthTest )
{
    reset();
    obj = makeDetached( text, depthTest );
    parent.addChild( obj );
}

std::shared_ptr<ObjectLabel> AncillaryLabel::makeDetached( const PositionedText& text, bool depthTest )
{
    std::shared_ptr<ObjectLabel> obj = std::make_shared<ObjectLabel>();
    obj->setAncillary( true );
    obj->setFrontColor( SceneColors::get( SceneColors::Type::Labels ), false );
    obj->setLabel( text );
    obj->setPickable( false );
    obj->setVisualizeProperty( depthTest, VisualizeMaskType::DepthTest, ViewportMask::all() );
    return obj;
}

void AncillaryLabel::reset()
{
    if ( obj )
        obj->detachFromParent();
    obj.reset();
}

void AncillaryLabel::setText( const PositionedText& text )
{
    obj->setLabel( text );
}

void AncillaryLabel::resetText()
{
    obj->setLabel( {} );
}

void AncillaryLabel::setDepthTest( bool depthTest )
{
    obj->setVisualizeProperty( depthTest, VisualizeMaskType::DepthTest, ViewportMask::all() );
}

void AncillaryLabel::setPosition( const Vector3f& pos )
{
    obj->setLabel( PositionedText{ obj->getLabel().text, pos } );
}

} //namespace MR
