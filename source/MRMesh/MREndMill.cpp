#include "MREndMill.h"

#include "MRAffineXf.h"
#include "MRConstants.h"
#include "MRCylinder.h"
#include "MRMakeSphereMesh.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRMeshFillHole.h"
#include "MRMeshTrimWithPlane.h"
#include "MRRegionBoundary.h"
#include "MRStringConvert.h"

#include "MRPch/MRJson.h"

namespace MR
{

Mesh EndMillTool::toMesh( int horizontalResolution, int verticalResolution ) const
{
    const auto radius = diameter / 2.f;

    switch ( cutter.type )
    {
    case EndMillCutter::Type::Flat:
        return makeCylinder( radius, length, horizontalResolution );

    case EndMillCutter::Type::Ball:
    {
        // TODO: custom implementation
        auto sphere = makeUVSphere( radius, horizontalResolution, verticalResolution );

        trimWithPlane( sphere, TrimWithPlaneParams {
            .plane = Plane3f{ Vector3f::minusZ(), 0.f },
        } );

        sphere.transform( AffineXf3f::translation( Vector3f::plusZ() * radius ) );

        const auto borderEdges = sphere.topology.findHoleRepresentiveEdges();
        assert( borderEdges.size() == 1 );
        auto borderEdge = borderEdges.front();
        borderEdge = makeDegenerateBandAroundHole( sphere, borderEdge );

        for ( auto e : trackRightBoundaryLoop( sphere.topology, borderEdge ) )
        {
            const auto v = sphere.topology.org( e );
            auto& p = sphere.points[v];
            p.z = length;
        }
        sphere.invalidateCaches();

        fillHole( sphere, borderEdge, {
            .metric = getPlaneFillMetric( sphere, borderEdge ),
        } );

        return sphere;
    }

    case EndMillCutter::Type::Count:
        MR_UNREACHABLE
    }
    MR_UNREACHABLE
}

void serializeToJson( const EndMillCutter& cutter, Json::Value& root )
{
    static const std::array knownValues {
        "flat",
        "ball",
    };
    static_assert( knownValues.size() == (int)EndMillCutter::Type::Count );
    root["Type"] = knownValues[(int)cutter.type];
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
    };
    assert( knownValues.size() == (int)EndMillCutter::Type::Count );
    if ( auto it = knownValues.find( toLower( root["Type"].asString() ) ); it != knownValues.end() )
        cutter.type = it->second;
    else
        return unexpected( fmt::format( "Invalid value: {}", "Type" ) );

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
