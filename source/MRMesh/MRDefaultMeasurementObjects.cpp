#include "MRDefaultMeasurementObjects.h"

#include "MRMesh/MRAngleMeasurementObject.h"
#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRDistanceMeasurementObject.h"
#include "MRMesh/MRRadiusMeasurementObject.h"
#include "MRMesh/MRSphereObject.h"

namespace MR
{

void attachDefaultMeasurementsToObject( Object& object, AttachDefaultMeasurementsFlags flags )
{
    auto addRadius = [&]( auto lambda )
    {
        if ( bool( flags & AttachDefaultMeasurementsFlags::Radiuses ) )
        {
            if ( auto radius = object.find<RadiusMeasurementObject>();
                bool( radius ) /*implies*/<= bool( flags & AttachDefaultMeasurementsFlags::Overwrite )
            )
            {
                if ( radius )
                    radius->detachFromParent();

                radius = std::make_shared<RadiusMeasurementObject>();
                object.addChild( radius );
                lambda( std::move( radius ) );
            }
        }
    };
    auto addDistance = [&]( auto lambda )
    {
        if ( bool( flags & AttachDefaultMeasurementsFlags::Distances ) )
        {
            if ( auto distance = object.find<DistanceMeasurementObject>();
                bool( distance ) /*implies*/<= bool( flags & AttachDefaultMeasurementsFlags::Overwrite )
            )
            {
                if ( distance )
                    distance->detachFromParent();

                distance = std::make_shared<DistanceMeasurementObject>();
                object.addChild( distance );
                lambda( std::move( distance ) );
            }
        }
    };
    auto addAngle = [&]( auto lambda )
    {
        if ( bool( flags & AttachDefaultMeasurementsFlags::Angles ) )
        {
            if ( auto angle = object.find<AngleMeasurementObject>();
                bool( angle ) /*implies*/<= bool( flags & AttachDefaultMeasurementsFlags::Overwrite )
            )
            {
                if ( angle )
                    angle->detachFromParent();

                angle = std::make_shared<AngleMeasurementObject>();
                object.addChild( angle );
                lambda( std::move( angle ) );
            }
        }
    };

    if ( dynamic_cast<CircleObject *>( &object ) )
    {
        addRadius( [&]( auto radius )
        {
            radius->setName( "Radius" );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( false );
        } );

        return;
    }

    if ( dynamic_cast<CylinderObject *>( &object ) )
    {
        addRadius( [&]( auto radius )
        {
            radius->setName( "Radius" );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( false );
        } );

        addDistance( [&]( auto height )
        {
            height->setName( "Distance" );
            height->setLocalPoint( Vector3f( 0, 0, -0.5f ) );
            height->setLocalDelta( Vector3f( 0, 0, 1 ) );
        } );

        return;
    }

    if ( dynamic_cast<ConeObject *>( &object ) )
    {
        addRadius( [&]( auto radius )
        {
            radius->setName( "Radius" );
            radius->setLocalCenter( Vector3f( 0, 0, 1 ) );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( false );
        } );

        addDistance( [&]( auto height )
        {
            height->setName( "Distance" );
            height->setLocalDelta( Vector3f( 0, 0, 1 ) );
        } );

        addAngle( [&]( auto angle )
        {
            angle->setName( "Angle" );
            angle->setLocalRays( Vector3f( 0.5f, 0, 0.5f ), Vector3f( -0.5f, 0, 0.5f ) );
            angle->setIsConical( true );
        } );

        return;
    }

    if ( dynamic_cast<SphereObject *>( &object ) )
    {
        addRadius( [&]( auto radius )
        {
            radius->setName( "Radius" );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( true );
        } );

        return;
    }
}

}
