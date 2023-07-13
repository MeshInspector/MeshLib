#include "MRCNCMachineSettings.h"

namespace MR
{

void CNCMachineSettings::setRotationParams( RotationAxisName paramName, const Vector3f& rotationAxis )
{
    const int intParamName = int( paramName );
    const bool validParamName = intParamName < int( RotationAxisName::Count );
    if ( !validParamName )
        return;

    if ( rotationAxis.lengthSq() < 0.01f )
        return;

    rotationAxes_[intParamName] = rotationAxis;
}

Vector3f CNCMachineSettings::getRotationParams( RotationAxisName paramName ) const
{
    const int intParamName = int( paramName );
    const bool validParamName = intParamName < int( RotationAxisName::Count );
    if ( !validParamName )
        return {};

    return rotationAxes_[intParamName];
}

void CNCMachineSettings::setRotationOrder( const RotationAxesOrder& rotationAxesOrder )
{
    rotationAxesOrder_.clear();
    for ( int i = 0; i < rotationAxesOrder.size(); ++i )
    {
        const int intRotationAxis = int( rotationAxesOrder_[i] );
        if ( intRotationAxis == int( CNCMachineSettings::RotationAxisName::Count ) )
            continue;

        if ( std::find( rotationAxesOrder_.begin(), rotationAxesOrder_.end(), rotationAxesOrder[i] ) != rotationAxesOrder_.end() )
            continue;

        rotationAxesOrder_.push_back( rotationAxesOrder[i] );
    }
}

}
