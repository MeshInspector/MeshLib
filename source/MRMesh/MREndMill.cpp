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

EndMillCutter EndMillCutter::makeFlat( float radius )
{
    assert( radius > 0.f );
    return {
        .type = Type::Flat,
        .radius = radius,
    };
}

EndMillCutter EndMillCutter::makeBall( float radius )
{
    assert( radius > 0.f );
    return {
        .type = Type::Ball,
        .radius = radius,
    };
}

Expected<EndMillCutter> EndMillCutter::deserialize( const Json::Value& root )
{
    EndMillCutter result;

    if ( root["Type"].type() != Json::stringValue )
        return unexpected( fmt::format( "Missing field: {}", "Type" ) );
    const auto type = toLower( root["Type"].asString() );
    static const HashMap<std::string, Type> knownValues {
        { "flat", Type::Flat },
        { "ball", Type::Ball },
    };
    assert( knownValues.size() == (int)Type::Count );
    if ( auto it = knownValues.find( type ); it != knownValues.end() )
        result.type = it->second;
    else
        return unexpected( fmt::format( "Invalid value: {}", "Type" ) );

    if ( root["Radius"].type() != Json::realValue )
        return unexpected( fmt::format( "Missing field: {}", "Radius" ) );
    result.radius = root["Radius"].asFloat();

    return result;
}

void EndMillCutter::serialize( Json::Value& root ) const
{
    static const std::array knownValues {
        "flat",
        "ball",
    };
    static_assert( knownValues.size() == (int)Type::Count );

    root["Type"] = knownValues[(int)type];
    root["Radius"] = radius;
}

Mesh EndMillTool::toMesh( float minEdgeLen ) const
{
    if ( minEdgeLen <= 0.f )
    {
        constexpr auto cResolution = 64;
        static const auto cChordLength = std::sin( PI_F / cResolution );
        minEdgeLen = cChordLength * 2.f * cutter.radius;
    }

    const auto resolution = (int)std::floor( 2.f * PI_F * cutter.radius / minEdgeLen );

    switch ( cutter.type )
    {
    case EndMillCutter::Type::Flat:
        return makeCylinder( cutter.radius, length, resolution );

    case EndMillCutter::Type::Ball:
    {
        // TODO: custom implementation
        auto sphere = makeUVSphere( cutter.radius, resolution, resolution );

        trimWithPlane( sphere, TrimWithPlaneParams {
            .plane = Plane3f{ Vector3f::minusZ(), 0.f },
        } );

        sphere.transform( AffineXf3f::translation( Vector3f::plusZ() * cutter.radius ) );

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

Expected<EndMillTool> EndMillTool::deserialize( const Json::Value& root )
{
    EndMillTool result;

    if ( root["Cutter"].type() != Json::objectValue )
        return unexpected( fmt::format( "Missing field: {}", "Cutter" ) );
    if ( auto res =  EndMillCutter::deserialize( root["Cutter"] ) )
        result.cutter = *res;
    else
        return unexpected( std::move( res.error() ) );

    if ( root["Length"].type() != Json::realValue )
        return unexpected( fmt::format( "Missing field: {}", "Length" ) );
    result.length = root["Length"].asFloat();

    return result;
}

void EndMillTool::serialize( Json::Value& root ) const
{
    root["Length"] = length;
    cutter.serialize( root["Cutter"] );
}

} // namespace MR
