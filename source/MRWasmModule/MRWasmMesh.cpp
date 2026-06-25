#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace emscripten;

namespace
{

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

}

EMSCRIPTEN_BINDINGS( meshlib_mesh )
{
    class_<MR::Mesh>( "Mesh" )
        .function( "numVerts", +[]( const MR::Mesh& m ) { return m.topology.numValidVerts(); } )
        .function( "numTris", +[]( const MR::Mesh& m ) { return m.topology.numValidFaces(); } )
        .function( "toGeometry", +[]( const MR::Mesh& m ) { return meshToGeometry( m, false ); } )
        .function( "toGeometryWithNormals", +[]( const MR::Mesh& m ) { return meshToGeometry( m, true ); } );

    function( "meshFromGeometry", &meshFromGeometry );
}
