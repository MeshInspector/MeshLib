#include "MRAncillaryPlane.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

AncillaryPlane& AncillaryPlane::operator =( AncillaryPlane&& b )
{
    reset(); obj = std::move( b.obj ); return *this;
}

AncillaryPlane::~AncillaryPlane()
{
    reset();
}
    
void AncillaryPlane::make()
{
    reset();
    obj = std::make_shared<PlaneObject>();
    obj->setAncillary( true );
    SceneRoot::get().addChild( obj );
}
 
void AncillaryPlane::reset()
{
    if ( obj )
        obj->detachFromParent();
    obj.reset();
}

}