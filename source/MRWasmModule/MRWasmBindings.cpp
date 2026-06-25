#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten.h>

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRBooleanOperation.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRMeshNormals.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace emscripten;

namespace
{

[[noreturn]] void throwJsError( const std::string& msg )
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( { throw new Error( UTF8ToString( $0 ) ); }, msg.c_str() );
#pragma clang diagnostic pop
    std::abort();
}

template <typename F>
auto guarded( F&& f ) -> decltype( f() )
{
    try
    {
        return f();
    }
    catch ( const std::exception& e )
    {
        throwJsError( e.what() );
    }
    catch ( ... )
    {
        throwJsError( "unknown error" );
    }
}

// Must copy out: a typed_memory_view aliases WASM memory and is detached when the
// heap grows, so it must never be retained or returned to JS.
val toFloat32Array( const float* data, size_t count )
{
    val out = val::global( "Float32Array" ).new_( count );
    if ( count != 0 )
        out.call<void>( "set", val( typed_memory_view( count, data ) ) );
    return out;
}

val toUint32Array( const uint32_t* data, size_t count )
{
    val out = val::global( "Uint32Array" ).new_( count );
    if ( count != 0 )
        out.call<void>( "set", val( typed_memory_view( count, data ) ) );
    return out;
}

MR::Mesh meshFromGeometry( val positions, val indices )
{
    return guarded( [&]() -> MR::Mesh
    {
        const std::vector<float> pos = convertJSArrayToNumberVector<float>( positions );
        const std::vector<uint32_t> idx = convertJSArrayToNumberVector<uint32_t>( indices );

        if ( pos.size() % 3 != 0 )
            throw std::runtime_error( "meshFromGeometry: positions length must be a multiple of 3" );
        if ( idx.size() % 3 != 0 )
            throw std::runtime_error( "meshFromGeometry: indices length must be a multiple of 3" );

        const size_t numVerts = pos.size() / 3;
        const size_t numTris = idx.size() / 3;

        MR::VertCoords coords;
        coords.vec_.resize( numVerts );
        if ( numVerts != 0 )
            std::memcpy( coords.vec_.data(), pos.data(), pos.size() * sizeof( float ) );

        MR::Triangulation tris;
        tris.vec_.resize( numTris );
        for ( size_t f = 0; f < numTris; ++f )
        {
            const uint32_t a = idx[3 * f + 0];
            const uint32_t b = idx[3 * f + 1];
            const uint32_t c = idx[3 * f + 2];
            if ( a >= numVerts || b >= numVerts || c >= numVerts )
                throw std::runtime_error( "meshFromGeometry: triangle vertex index out of range" );
            tris.vec_[f] = MR::ThreeVertIds{ MR::VertId( (int)a ), MR::VertId( (int)b ), MR::VertId( (int)c ) };
        }

        return MR::Mesh::fromTriangles( std::move( coords ), tris );
    } );
}

val meshToGeometry( const MR::Mesh& meshIn, bool wantNormals )
{
    return guarded( [&]() -> val
    {
        MR::Mesh mesh = meshIn;
        mesh.pack();

        const size_t numVerts = mesh.points.size();
        val positions = toFloat32Array(
            reinterpret_cast<const float*>( mesh.points.vec_.data() ), numVerts * 3 );

        const std::vector<MR::ThreeVertIds> tris = mesh.topology.getAllTriVerts();
        std::vector<uint32_t> flatIdx( tris.size() * 3 );
        for ( size_t f = 0; f < tris.size(); ++f )
        {
            flatIdx[3 * f + 0] = (uint32_t)(int)tris[f][0];
            flatIdx[3 * f + 1] = (uint32_t)(int)tris[f][1];
            flatIdx[3 * f + 2] = (uint32_t)(int)tris[f][2];
        }
        val indices = toUint32Array( flatIdx.data(), flatIdx.size() );

        val result = val::object();
        result.set( "positions", positions );
        result.set( "indices", indices );

        if ( wantNormals )
        {
            const MR::VertNormals normals = MR::computePerVertNormals( mesh );
            result.set( "normals", toFloat32Array(
                reinterpret_cast<const float*>( normals.vec_.data() ), normals.size() * 3 ) );
        }
        return result;
    } );
}

enum class JsBooleanOp { Union, Intersection, DifferenceAB, DifferenceBA };

MR::BooleanOperation toCoreOp( JsBooleanOp op )
{
    switch ( op )
    {
    case JsBooleanOp::Union:        return MR::BooleanOperation::Union;
    case JsBooleanOp::Intersection: return MR::BooleanOperation::Intersection;
    case JsBooleanOp::DifferenceAB: return MR::BooleanOperation::DifferenceAB;
    case JsBooleanOp::DifferenceBA: return MR::BooleanOperation::DifferenceBA;
    }
    throw std::runtime_error( "boolean: unknown operation" );
}

MR::Mesh booleanOp( const MR::Mesh& a, const MR::Mesh& b, JsBooleanOp op )
{
    return guarded( [&]() -> MR::Mesh
    {
        MR::BooleanResult res = MR::boolean( a, b, toCoreOp( op ) );
        if ( !res.valid() )
            throw std::runtime_error( res.errorString.empty() ? "boolean operation failed" : res.errorString );
        return std::move( res.mesh );
    } );
}

struct DecimateResultJs
{
    int vertsDeleted = 0;
    int facesDeleted = 0;
    float errorIntroduced = 0;
    bool cancelled = false;
};

DecimateResultJs decimate( MR::Mesh& mesh, val settings )
{
    return guarded( [&]() -> DecimateResultJs
    {
        MR::DecimateSettings s;
        s.packMesh = true;

        const int curFaces = mesh.topology.numValidFaces();
        const bool hasSettings = !settings.isUndefined() && !settings.isNull();
        auto has = [&]( const char* k ) { return hasSettings && settings.hasOwnProperty( k ); };
        auto num = [&]( const char* k ) { return settings[k].as<double>(); };

        bool haveStop = false;

        if ( has( "targetTriangleCount" ) )
        {
            int target = (int)num( "targetTriangleCount" );
            if ( target < 0 )
                target = 0;
            s.maxDeletedFaces = std::max( 0, curFaces - target );
            haveStop = true;
        }
        else if ( has( "targetRatio" ) )
        {
            const double r = std::clamp( num( "targetRatio" ), 0.0, 1.0 );
            const int target = (int)std::llround( curFaces * r );
            s.maxDeletedFaces = std::max( 0, curFaces - target );
            haveStop = true;
        }

        if ( has( "maxDeletedFaces" ) ) { s.maxDeletedFaces = (int)num( "maxDeletedFaces" ); haveStop = true; }
        if ( has( "maxError" ) ) { s.maxError = (float)num( "maxError" ); haveStop = true; }
        if ( has( "maxEdgeLen" ) ) { s.maxEdgeLen = (float)num( "maxEdgeLen" ); haveStop = true; }
        if ( has( "strategy" ) )
        {
            const std::string st = settings["strategy"].as<std::string>();
            s.strategy = ( st == "shortestEdgeFirst" )
                ? MR::DecimateStrategy::ShortestEdgeFirst
                : MR::DecimateStrategy::MinimizeError;
        }

        if ( !haveStop )
            throw std::runtime_error( "decimate: provide a stopping criterion "
                "(targetRatio, targetTriangleCount, maxDeletedFaces, maxError or maxEdgeLen)" );

        const MR::DecimateResult r = MR::decimateMesh( mesh, s );
        return DecimateResultJs{ r.vertsDeleted, r.facesDeleted, r.errorIntroduced, r.cancelled };
    } );
}

}

EMSCRIPTEN_BINDINGS( meshlib )
{
    enum_<JsBooleanOp>( "BooleanOp" )
        .value( "Union", JsBooleanOp::Union )
        .value( "Intersection", JsBooleanOp::Intersection )
        .value( "DifferenceAB", JsBooleanOp::DifferenceAB )
        .value( "DifferenceBA", JsBooleanOp::DifferenceBA );

    value_object<DecimateResultJs>( "DecimateResult" )
        .field( "vertsDeleted", &DecimateResultJs::vertsDeleted )
        .field( "facesDeleted", &DecimateResultJs::facesDeleted )
        .field( "errorIntroduced", &DecimateResultJs::errorIntroduced )
        .field( "cancelled", &DecimateResultJs::cancelled );

    class_<MR::Mesh>( "Mesh" )
        .function( "numVerts", +[]( const MR::Mesh& m ) { return m.topology.numValidVerts(); } )
        .function( "numTris", +[]( const MR::Mesh& m ) { return m.topology.numValidFaces(); } )
        .function( "toGeometry", +[]( const MR::Mesh& m ) { return meshToGeometry( m, false ); } )
        .function( "toGeometryWithNormals", +[]( const MR::Mesh& m ) { return meshToGeometry( m, true ); } );

    function( "meshFromGeometry", &meshFromGeometry );
    function( "boolean", &booleanOp );
    function( "decimate", &decimate );
}

int main()
{
    return 0;
}
