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
#include <memory>
#include <vector>

using namespace emscripten;

namespace
{

std::shared_ptr<MR::Mesh> meshFromGeometry( val positions, val indices )
{
    const std::vector<float> pos = convertJSArrayToNumberVector<float>( positions );
    const std::vector<uint32_t> idx = convertJSArrayToNumberVector<uint32_t>( indices );

    const size_t numVerts = pos.size() / 3;
    const size_t numTris = idx.size() / 3;

    MR::VertCoords coords;
    coords.vec_.resize( numVerts );
    if ( numVerts != 0 )
        std::memcpy( coords.vec_.data(), pos.data(), numVerts * 3 * sizeof( float ) );

    MR::Triangulation tris;
    tris.vec_.resize( numTris );
    for ( size_t f = 0; f < numTris; ++f )
        tris.vec_[f] = MR::ThreeVertIds{
            MR::VertId( (int)idx[3 * f + 0] ),
            MR::VertId( (int)idx[3 * f + 1] ),
            MR::VertId( (int)idx[3 * f + 2] ) };

    return std::make_shared<MR::Mesh>( MR::Mesh::fromTriangles( std::move( coords ), tris ) );
}

val meshToGeometry( std::shared_ptr<MR::Mesh> meshIn, bool wantNormals )
{
    MR::Mesh mesh = *meshIn;
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
}

}

EMSCRIPTEN_BINDINGS( meshlib_mesh )
{
    class_<MR::Mesh>( "Mesh" )
        .smart_ptr<std::shared_ptr<MR::Mesh>>( "MeshPtr" );

    function( "meshFromGeometry", &meshFromGeometry );
    function( "meshToGeometry", &meshToGeometry );
}
