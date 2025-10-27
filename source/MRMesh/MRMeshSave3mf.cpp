#include "MRMeshSave.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOFilters.h"
#include "MRStringConvert.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

namespace MeshSave
{

Expected<void> toModel3mf( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( mesh, out, settings );
}

Expected<void> toModel3mf( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = mesh.topology.lastValidVert();
    const bool saveColors = settings.colors && settings.colors->size() > lastVertId;

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n"
        "element vertex " << numPoints << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( saveColors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";

    const auto fLast = mesh.topology.lastValidFace();
    const auto numSaveFaces = settings.packPrimitives ? mesh.topology.numValidFaces() : int( fLast + 1 );
    out <<  "element face " << numSaveFaces << "\nproperty list uchar int vertex_indices\nend_header\n";

    static_assert( sizeof( Vector3f ) == 12, "wrong size of Vector3f" );
#pragma pack(push, 1)
    struct PlyColor
    {
        unsigned char r = 0, g = 0, b = 0;
    };
#pragma pack(pop)
    static_assert( sizeof( PlyColor ) == 3, "check your padding" );

    // write vertices
    int numSaved = 0;
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !mesh.topology.hasVert( i ) )
            continue;
        const Vector3f p = applyFloat( settings.xf, mesh.points[i] );
        out.write( ( const char* )&p, 12 );
        if ( settings.colors )
        {
            const auto c = ( *settings.colors )[i];
            PlyColor pc{ .r = c.r, .g = c.g, .b = c.b };
            out.write( ( const char* )&pc, 3 );
        }
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / numPoints * 0.5f ) )
            return unexpectedOperationCanceled();
    }

    // write triangles
    #pragma pack(push, 1)
    struct PlyTriangle
    {
        char cnt = 3;
        int v[3];
    };
    #pragma pack(pop)
    static_assert( sizeof( PlyTriangle ) == 13, "check your padding" );

    PlyTriangle tri;
    int savedFaces = 0;
    for ( FaceId f{0}; f <= fLast; ++f )
    {
        if ( mesh.topology.hasFace( f ) )
        {
            VertId vs[3];
            mesh.topology.getTriVerts( f, vs );
            for ( int i = 0; i < 3; ++i )
                tri.v[i] = vertRenumber( vs[i] );
        }
        else if ( !settings.packPrimitives )
            tri.v[0] = tri.v[1] = tri.v[2] = 0;
        else
            continue;
        out.write( (const char *)&tri, sizeof( PlyTriangle ) );
        ++savedFaces;
        if ( settings.progress && !( savedFaces & 0x3FF ) && !settings.progress( float( savedFaces ) / numSaveFaces * 0.5f + 0.5f ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PLY-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

MR_ADD_MESH_SAVER( IOFilter( "3D Manufacturing model (.model)", "*.model" ), toModel3mf, {} )

} //namespace MeshSave

} //namespace MR
