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

void attachDefaultMeasurementsToObject( Object& object, const AttachDefaultMeasurementsParams& params )
{
    auto addBasic = [&]( auto typeTag, DefaultMeasurementKinds bit, auto lambda )
    {
        using T = typename decltype(typeTag)::type;

        if ( bool( params.enabledKinds & bit ) )
        {
            if ( auto measurement = object.find<T>(); bool( measurement ) /*implies*/<= params.overwrite )
            {
                if ( measurement )
                    measurement->detachFromParent();

                measurement = std::make_shared<T>();
                object.addChild( measurement );
                measurement->setVisible( bool( params.defaultVisibleKinds & bit ) );
                lambda( std::move( measurement ) );
            }
        }
    };

    auto addRadius = [&]( auto lambda ) { addBasic( std::type_identity<RadiusMeasurementObject>{}, DefaultMeasurementKinds::Radiuses, std::move( lambda ) ); };
    auto addDistance = [&]( auto lambda ) { addBasic( std::type_identity<DistanceMeasurementObject>{}, DefaultMeasurementKinds::Distances, std::move( lambda ) ); };
    auto addAngle = [&]( auto lambda ) { addBasic( std::type_identity<AngleMeasurementObject>{}, DefaultMeasurementKinds::Angles, std::move( lambda ) ); };

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
            height->setName( "Height" );
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
            height->setName( "Height" );
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
