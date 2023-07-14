#include "MRCNCMachineSettings.h"

namespace MR
{

void CNCMachineSettings::setRotationAxis( RotationAxisName paramName, const Vector3f& rotationAxis )
{
    const int intParamName = int( paramName );
    if ( rotationAxis.lengthSq() == 0.f )
        return;

    rotationAxes_[intParamName] = rotationAxis.normalized();
}

const Vector3f& CNCMachineSettings::getRotationAxis( RotationAxisName paramName ) const
{
    const int intParamName = int( paramName );
    return rotationAxes_[intParamName];
}

void CNCMachineSettings::setRotationOrder( const RotationAxesOrder& rotationAxesOrder )
{
    rotationAxesOrder_.clear();
    for ( int i = 0; i < rotationAxesOrder.size(); ++i )
    {
        if ( std::find( rotationAxesOrder_.begin(), rotationAxesOrder_.end(), rotationAxesOrder[i] ) != rotationAxesOrder_.end() )
            continue;

        rotationAxesOrder_.push_back( rotationAxesOrder[i] );
    }
}

}
