#pragma once

#include "MRPointOnObject.h"
#include "MRObjectMeshHolder.h"
#include "MRObjectPointsHolder.h"
#include "MRMesh.h"

namespace MR
{
PickedPoint pointOnObject2PickedPoint( const VisualObject* surface, const PointOnObject& pos )
{
    const ObjectMeshHolder* objMesh = dynamic_cast< const ObjectMeshHolder* >( surface );
    if ( objMesh )
    {
        return objMesh->mesh()->toTriPoint( pos );
    }
    const ObjectPointsHolder* objPoints = dynamic_cast< const ObjectPointsHolder* >( surface );
    if ( objPoints )
    {
        return pos.vert;
    }

    //////// TO DO  //////
    /*
    const ObjectLines* objLines = dynamic_cast< const ObjectLines* >( surface );
    if ( objPoints )
    {
        return -1; // TODO support Lines
    }
    */

    return -1;
}

MR::Vector3f getPickedPointCenter3D( const VisualObject* surface, const PickedPoint& point )
{

    if ( const MeshTriPoint* triPoint = std::get_if<MeshTriPoint>( &point ) )
    {
        const ObjectMeshHolder* objMesh = dynamic_cast< const ObjectMeshHolder* >( surface );
        assert( objMesh );
        if ( objMesh )
        {
            return objMesh->mesh()->triPoint( *triPoint );
        }
    }
    else if ( const EdgePoint* edgePoint = std::get_if<EdgePoint>( &point ) )
    {
        // TODO  Handle EdgePoint
        // ...
    }
    else if ( const VertId* vertId = std::get_if<VertId>( &point ) )
    {
        // TODO  Handle VertId
        // ...
    }
    else if ( const int* intValue = std::get_if<int>( &point ) )
    {
        //  TODO Handle int
        // ...
    }

    assert( false );
    return {};
}

} //namespace MR
