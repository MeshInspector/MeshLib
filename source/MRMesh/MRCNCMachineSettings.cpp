#include "MRCNCMachineSettings.h"
#include "MRSerializer.h"

namespace MR
{
const std::string axesName[] = { "Axis A", "Axis B", "Axis C" };

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

void CNCMachineSettings::setFeedrateIdle( float feedrateIdle )
{
    feedrateIdle_ = std::clamp( feedrateIdle, 0.f, 100000.f );
}

Json::Value CNCMachineSettings::saveToJson() const
{
    Json::Value jsonValue;
    std::string orderStr;
    Vector3b activeAxes;
    for ( int i = 0; i < rotationAxesOrder_.size(); ++i )
    {
        if ( rotationAxesOrder_[i] == RotationAxisName::A )
        {
            orderStr += "A";
            activeAxes.x = true;
        }
        else if ( rotationAxesOrder_[i] == RotationAxisName::B )
        {
            orderStr += "B";
            activeAxes.y = true;
        }
        else if ( rotationAxesOrder_[i] == RotationAxisName::C )
        {
            orderStr += "C";
            activeAxes.z = true;
        }
    }
    jsonValue["Axes Order"] = orderStr;
    for ( int i = 0; i < 3; ++i )
    {
        if ( activeAxes[i] )
            serializeToJson( rotationAxes_[i], jsonValue[axesName[i]] );
    }
    jsonValue["Feedrate Idle"] = feedrateIdle_;
    return jsonValue;
}

bool CNCMachineSettings::loadFromJson( const Json::Value& jsonValue )
{
    Vector3b readedAxes;
    if ( jsonValue["Axes Order"].isString() )
    {
        RotationAxesOrder rotationAxesOrder;
        std::string orderStr = jsonValue["Axes Order"].asString();
        for ( int i = 0; i < orderStr.size(); ++i )
        {
            if ( orderStr[i] == 'A' )
            {
                if ( readedAxes.x )
                    return false;
                readedAxes.x = true;
                rotationAxesOrder.push_back( RotationAxisName::A );
            }
            else if ( orderStr[i] == 'B' )
            {
                if ( readedAxes.y )
                    return false;
                readedAxes.y = true;
                rotationAxesOrder.push_back( RotationAxisName::B );
            }
            else if ( orderStr[i] == 'C' )
            {
                if ( readedAxes.z )
                    return false;
                readedAxes.z = true;
                rotationAxesOrder.push_back( RotationAxisName::C );
            }
        }
        setRotationOrder( rotationAxesOrder );
    }
    else
        return false;

    auto loadAxis = [&] ( const std::string& jsonName, RotationAxisName axisName )
    {
        Vector3f vec3f;
        deserializeFromJson( jsonValue[jsonName], vec3f );
        if ( vec3f == Vector3f() )
            return false;
        setRotationAxis( axisName, vec3f );
        return true;
    };
    
    for ( int i = 0; i < 3; ++i )
    {
        if ( readedAxes[i] && !loadAxis( axesName[i], RotationAxisName( i ) ) )
            return false;
    }

    if ( jsonValue["Feedrate Idle"].isDouble() )
        feedrateIdle_ = jsonValue["Feedrate Idle"].asFloat();
    else
        return false;

    return true;
}

}
