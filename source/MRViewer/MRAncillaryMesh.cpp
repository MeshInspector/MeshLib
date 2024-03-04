#include "MRAncillaryMesh.h"
#include "MRMesh/MRObjectMesh.h"

namespace MR
{

void AncillaryMesh::make( Object& parent )
{
    reset();
    obj = std::make_shared<ObjectMesh>();
    obj->setAncillary( true );
    obj->setPickable( false );
    parent.addChild( obj );
}

void AncillaryMesh::reset()
{
    if ( obj )
        obj->detachFromParent();
    obj.reset();
}

} //namespace MR
