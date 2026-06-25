#include "MRMeshOrPointsObject.h"
#include "MRVisualObject.h"
#include "MRObjectMesh.h"
#include "MRObjectPoints.h"
#include "MRMeshOrPoints.h"


namespace MR
{

MeshOrPointsObject::MeshOrPointsObject( std::shared_ptr<VisualObject> vo )
{
    if ( auto objMesh = std::dynamic_pointer_cast< ObjectMesh >( std::move( vo ) ) )
        set( std::move( objMesh ) );
    else if ( auto objPoints = std::dynamic_pointer_cast< ObjectPoints >( std::move( vo ) ) )
        set( std::move( objPoints ) );
    else
        reset();
}

void MeshOrPointsObject::set( std::shared_ptr<ObjectMesh> om )
{
    var_ = om.get();
    visualObject_ = std::move( om );
}

ObjectMesh* MeshOrPointsObject::asObjectMesh() const
{
    if ( std::holds_alternative<ObjectMesh*>( var_ ) )
        return std::get<ObjectMesh*>( var_ );
    return {};
}

void MeshOrPointsObject::set( std::shared_ptr<ObjectPoints> op )
{
    var_ = op.get();
    visualObject_ = std::move( op );
}

ObjectPoints* MeshOrPointsObject::asObjectPoints() const
{
    if ( std::holds_alternative<ObjectPoints*>( var_ ) )
        return std::get<ObjectPoints*>( var_ );
    return {};
}

MeshOrPoints MeshOrPointsObject::meshOrPoints() const
{
    return std::visit( overloaded{
        [&]( ObjectMesh* objMesh )
        {
            return MeshOrPoints( objMesh->meshPart() );
        },
        [&] ( ObjectPoints* objPnts )
        {
            return MeshOrPoints( objPnts->pointCloudPart() );
        }
    }, var_ );
}

}
