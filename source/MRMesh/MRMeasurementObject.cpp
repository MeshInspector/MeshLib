#include "MRMeasurementObject.h"

#include "MRSceneColors.h"

namespace MR
{

MeasurementObject::MeasurementObject()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedMeasurements ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedMeasurements ), false );
}

}
