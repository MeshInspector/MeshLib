#include "MRDefaultMeasurementObjects.h"

#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRRadiusMeasurementObject.h"
#include "MRMesh/MRSphereObject.h"

namespace MR
{

void attachDefaultMeasurementsToObject( Object& object, const AttachDefaultMeasurementsParams& params )
{
    if ( dynamic_cast<CircleObject *>( &object ) )
    {
        if ( auto radius = object.find<RadiusMeasurementObject>();
            bool( radius ) /*implies*/<= params.overwrite
        )
        {
            if ( radius )
                radius->detachFromParent();

            radius = std::make_shared<RadiusMeasurementObject>();
            object.addChild( radius );
            radius->setName( "Radius" );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( false );
        }

        return;
    }

    if ( dynamic_cast<CylinderObject *>( &object ) )
    {
        if ( auto radius = object.find<RadiusMeasurementObject>();
            bool( radius ) /*implies*/<= params.overwrite
        )
        {
            if ( radius )
                radius->detachFromParent();

            radius = std::make_shared<RadiusMeasurementObject>();
            object.addChild( radius );
            radius->setName( "Radius" );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( false );
        }

        return;
    }

    if ( dynamic_cast<ConeObject *>( &object ) )
    {
        if ( auto radius = object.find<RadiusMeasurementObject>();
            bool( radius ) /*implies*/<= params.overwrite
        )
        {
            if ( radius )
                radius->detachFromParent();

            radius = std::make_shared<RadiusMeasurementObject>();
            object.addChild( radius );
            radius->setName( "Radius" );
            radius->setLocalCenter( Vector3f( 0, 0, 1 ) );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( false );
        }

        return;
    }

    if ( dynamic_cast<SphereObject *>( &object ) )
    {
        if ( auto radius = object.find<RadiusMeasurementObject>();
            bool( radius ) /*implies*/<= params.overwrite
        )
        {
            if ( radius )
                radius->detachFromParent();

            radius = std::make_shared<RadiusMeasurementObject>();
            object.addChild( radius );
            radius->setName( "Radius" );
            radius->setDrawAsDiameter( true );
            radius->setIsSpherical( true );
        }

        return;
    }
}

}
