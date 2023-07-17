#include "MRCNCMachineSettings.h"
#include "MRSerializer.h"

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

Json::Value CNCMachineSettings::saveToJson() const
{
    Json::Value jsonValue;
    serializeToJson( rotationAxes_[0], jsonValue["Axis A"] );
    serializeToJson( rotationAxes_[1], jsonValue["Axis B"] );
    serializeToJson( rotationAxes_[2], jsonValue["Axis C"] );
    std::string orderStr;
    for ( int i = 0; i < orderStr.size(); ++i )
    {
        if ( rotationAxesOrder_[i] == RotationAxisName::A )
            orderStr += "A";
        else if ( rotationAxesOrder_[i] == RotationAxisName::B )
            orderStr += "B";
        else if ( rotationAxesOrder_[i] == RotationAxisName::C )
            orderStr += "C";
    }
    jsonValue["Axes Order"] = orderStr;
    return jsonValue;
}

void CNCMachineSettings::loadFromJson( const Json::Value& jsonValue )
{
    auto loadAxis = [&] ( const std::string& jsonName, RotationAxisName axisName )
    {
        Vector3f vec3f;
        deserializeFromJson( jsonValue[jsonName], vec3f );
        setRotationAxis( axisName, vec3f );
    };
    loadAxis( "Axis A", RotationAxisName::A );
    loadAxis( "Axis B", RotationAxisName::B );
    loadAxis( "Axis C", RotationAxisName::C );
    if ( jsonValue["Axes Order"].isString() )
    {
        RotationAxesOrder rotationAxesOrder;
        std::string orderStr = jsonValue["Axes Order"].asString();
        for ( int i = 0; i < orderStr.size(); ++i )
        {
            if ( orderStr[i] == 'A' )
                rotationAxesOrder.push_back( RotationAxisName::A );
            else if ( orderStr[i] == 'B' )
                rotationAxesOrder.push_back( RotationAxisName::B );
            else if ( orderStr[i] == 'C' )
                rotationAxesOrder.push_back( RotationAxisName::C );
        }
        setRotationOrder( rotationAxesOrder );
    }
}

}
