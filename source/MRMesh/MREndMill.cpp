#include "MREndMill.h"

#include "MRConstants.h"
#include "MRMesh.h"
#include "MRSolidOfRevolution.h"
#include "MRStringConvert.h"
#include "MRVector2.h"

#include "MRPch/MRFmt.h"
#include "MRPch/MRJson.h"

namespace MR
{

float EndMillTool::getMinimalCutLength() const
{
    switch ( cutter.type )
    {
    case EndMillCutter::Type::Flat:
        return 0.f;
    case EndMillCutter::Type::Ball:
        return diameter / 2.f;
    case EndMillCutter::Type::BullNose:
        return cutter.cornerRadius;
    case EndMillCutter::Type::Chamfer:
        return std::tan( ( 90.f - cutter.cuttingAngle / 2.f ) * PI_F / 180.f ) * ( diameter - cutter.endDiameter ) / 2.f;
    case EndMillCutter::Type::Count:
        MR_UNREACHABLE
    }
    MR_UNREACHABLE
}

Mesh EndMillTool::toMesh( int horizontalResolution, int verticalResolution ) const
{
    const auto radius = diameter / 2.f;

    Contour2f profile;
    profile.emplace_back( 0.f, 0.f );

    switch ( cutter.type )
    {
    case EndMillCutter::Type::Flat:
        break;

    case EndMillCutter::Type::Ball:
        for ( auto i = 1; i < verticalResolution; ++i )
        {
            const auto angle = PI2_F * (float)i / (float)verticalResolution;
            profile.emplace_back(
                radius * std::sin( angle ),
                radius * ( 1 - std::cos( angle ) )
            );
        }
        break;

    case EndMillCutter::Type::BullNose:
        for ( auto i = 1; i < verticalResolution; ++i )
        {
            const auto angle = PI2_F * (float)i / (float)verticalResolution;
            profile.emplace_back(
                radius - cutter.cornerRadius * ( 1 - std::sin( angle ) ),
                cutter.cornerRadius * ( 1 - std::cos( angle ) )
            );
        }
        break;

    case EndMillCutter::Type::Chamfer:
        if ( cutter.endDiameter > 0.f )
            profile.emplace_back( cutter.endDiameter / 2.f, 0.f );
        break;

    case EndMillCutter::Type::Count:
        MR_UNREACHABLE
    }

    profile.emplace_back( radius, getMinimalCutLength() );
    profile.emplace_back( radius, length );
    profile.emplace_back( 0.f, length );

    return makeSolidOfRevolution( profile, horizontalResolution );
}

void serializeToJson( const EndMillCutter& cutter, Json::Value& root )
{
    static const std::array knownValues {
        "flat",
        "ball",
        "bull-nose",
        "chamfer",
    };
    static_assert( knownValues.size() == (int)EndMillCutter::Type::Count );
    root["Type"] = knownValues[(int)cutter.type];

    if ( cutter.type == EndMillCutter::Type::BullNose )
        root["CornerRadius"] = cutter.cornerRadius;

    if ( cutter.type == EndMillCutter::Type::Chamfer )
    {
        root["CuttingAngle"] = cutter.cuttingAngle;
        if ( cutter.endDiameter > 0.f )
            root["EndDiameter"] = cutter.endDiameter;
    }
}

void serializeToJson( const EndMillTool& tool, Json::Value& root )
{
    root["Length"] = tool.length;
    root["Diameter"] = tool.diameter;
    serializeToJson( tool.cutter, root["Cutter"] );
}

Expected<void> deserializeFromJson( const Json::Value& root, EndMillCutter& cutter )
{
    if ( root["Type"].type() != Json::stringValue )
        return unexpected( fmt::format( "Missing field: {}", "Type" ) );
    static const HashMap<std::string, EndMillCutter::Type> knownValues {
        { "flat", EndMillCutter::Type::Flat },
        { "ball", EndMillCutter::Type::Ball },
        { "bull-nose", EndMillCutter::Type::BullNose },
        { "chamfer", EndMillCutter::Type::Chamfer },
    };
    assert( knownValues.size() == (int)EndMillCutter::Type::Count );
    if ( auto it = knownValues.find( toLower( root["Type"].asString() ) ); it != knownValues.end() )
        cutter.type = it->second;
    else
        return unexpected( fmt::format( "Invalid value: {}", "Type" ) );

    if ( cutter.type == EndMillCutter::Type::BullNose )
    {
        if ( root["CornerRadius"].type() != Json::realValue )
            return unexpected( fmt::format( "Missing field: {}", "CornerRadius" ) );
        cutter.cornerRadius = root["CornerRadius"].asFloat();
    }

    if ( cutter.type == EndMillCutter::Type::Chamfer )
    {
        if ( root["CuttingAngle"].type() != Json::realValue )
            return unexpected( fmt::format( "Missing field: {}", "CuttingAngle" ) );
        cutter.cuttingAngle = root["CuttingAngle"].asFloat();

        if ( root["EndDiameter"].type() == Json::realValue )
            cutter.endDiameter = root["EndDiameter"].asFloat();
    }

    return {};
}

Expected<void> deserializeFromJson( const Json::Value& root, EndMillTool& tool )
{
    if ( root["Cutter"].type() != Json::objectValue )
        return unexpected( fmt::format( "Missing field: {}", "Cutter" ) );
    if ( auto res = deserializeFromJson( root["Cutter"], tool.cutter ); !res )
        return unexpected( std::move( res.error() ) );

    if ( root["Length"].type() != Json::realValue )
        return unexpected( fmt::format( "Missing field: {}", "Length" ) );
    tool.length = root["Length"].asFloat();

    if ( root["Diameter"].type() != Json::realValue )
        return unexpected( fmt::format( "Missing field: {}", "Diameter" ) );
    tool.diameter = root["Diameter"].asFloat();

    return {};
}

} // namespace MR
