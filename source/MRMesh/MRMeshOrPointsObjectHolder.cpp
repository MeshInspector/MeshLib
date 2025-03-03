#include "MRMeshOrPointsObjectHolder.h"
#include "MRVisualObject.h"
#include "MRObjectMesh.h"
#include "MRObjectPoints.h"


namespace MR
{



MeshOrPointsObjectHolder::MeshOrPointsObjectHolder( std::shared_ptr<VisualObject> vo )
{
    if ( auto objMesh = std::dynamic_pointer_cast< ObjectMesh >( vo ) )
        set( objMesh );
    else if ( auto objPoints = std::dynamic_pointer_cast< ObjectPoints >( vo ) )
        set( objPoints );
    else
        reset();
}

void MeshOrPointsObjectHolder::set( std::shared_ptr<ObjectMesh> om )
{
    var_ = om.get();
    visualObject_ = std::move( om );
}

ObjectMesh* MeshOrPointsObjectHolder::asObjectMesh() const
{
    if ( std::holds_alternative<ObjectMesh*>( var_ ) )
        return std::get<ObjectMesh*>( var_ );
    return {};
}

void MeshOrPointsObjectHolder::set( std::shared_ptr<ObjectPoints> op )
{
    var_ = op.get();
    visualObject_ = std::move( op );
}

ObjectPoints* MeshOrPointsObjectHolder::asObjectPoints() const
{
    if ( std::holds_alternative<ObjectPoints*>( var_ ) )
        return std::get<ObjectPoints*>( var_ );
    return {};
}

MeshOrPoints MeshOrPointsObjectHolder::meshOrPoints() const
{
    return std::visit( overloaded{
        [&]( ObjectMesh* objMesh )
        {
            return MeshOrPoints( objMesh->meshPart() );
        },
        [&] ( ObjectPoints* objPnts )
        {
            return MeshOrPoints( *objPnts->pointCloud() );
        }
    }, var_ );
}

}
