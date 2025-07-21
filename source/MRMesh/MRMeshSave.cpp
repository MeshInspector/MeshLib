#include "MRMeshSave.h"
#include "MRIOFormatsRegistry.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRBitSetParallelFor.h"
#include "MRPch/MRFmt.h"
#include "MRMeshTexture.h"
#include "MRImageSave.h"

namespace MR
{

namespace MeshSave
{

Expected<void> toMrmesh( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrmesh( mesh, out, settings );
}

Expected<void> toMrmesh( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;
    mesh.topology.write( out );

    // write points
    auto numPoints = (std::uint32_t)( mesh.topology.lastValidVert() + 1 );
    out.write( (const char*)&numPoints, 4 );

    VertCoords buf;
    const auto & xfVerts = transformPoints( mesh.points, mesh.topology.getValidVerts(), settings.xf, buf );
    if ( !writeByBlocks( out, ( const char* )xfVerts.data(), numPoints * sizeof( Vector3f ), settings.progress ) )
        return unexpectedOperationCanceled();

    if ( !out )
        return unexpected( std::string( "Error saving in Mrmesh-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toOff( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    // although .off is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toOff( mesh, out, settings );
}

Expected<void> toOff( const Mesh& mesh, std::ostream& out, const SaveSettings & settings )
{
    MR_TIMER;

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = mesh.topology.lastValidVert();
    const int numPolygons = mesh.topology.numValidFaces();

    out << "OFF\n" << numPoints << ' ' << numPolygons << " 0\n\n";
    int numSaved = 0;
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !mesh.topology.hasVert( i ) )
            continue;
        auto saveVertex = [&]( auto && p )
        {
            out << fmt::format( "{} {} {}\n", p.x, p.y, p.z );
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, mesh.points[i] ) );
        else
            saveVertex( mesh.points[i] );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / numPoints * 0.5f ) )
            return unexpectedOperationCanceled();
    }
    out << '\n';

    const float facesNum = float( mesh.topology.edgePerFace().size() );
    size_t faceIndex = 0;
    for ( const auto& e : mesh.topology.edgePerFace() )
    {
        ++faceIndex;
        if ( settings.progress && !( faceIndex & 0x3FF ) && !settings.progress( float( faceIndex ) / facesNum * 0.5f + 0.5f ) )
            return unexpectedOperationCanceled();
        if ( !e.valid() )
            continue;

        VertId a, b, c;
        mesh.topology.getLeftTriVerts( e, a, b, c );
        out << fmt::format( "3 {} {} {}\n", vertRenumber( a ), vertRenumber( b ), vertRenumber( c ) );
    }

    if ( !out )
        return unexpected( std::string( "Error saving in OFF-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}


Expected<void> toObj( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, int firstVertId )
{
    // although .obj is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

#ifndef __EMSCRIPTEN__
    // it is hard to handle several files output for browser, so for now it is under ifdef,
    // anyway later it may be reworked to save simple zip and taken out of ifdef
    if ( settings.uvMap )
    {
        if ( auto pngSaver = ImageSave::getImageSaver( "*.png" ) )
        {
            auto mtlPath = file.parent_path() / ( settings.materialName + ".mtl" );
            std::ofstream ofMtl( mtlPath, std::ofstream::binary );
            if ( ofMtl )
            {
                ofMtl << "newmtl Texture\n";
                if ( settings.texture && pngSaver( *settings.texture, file.parent_path() / ( settings.materialName + ".png" ) ).has_value() )
                    ofMtl << fmt::format( "map_Kd {}\n", settings.materialName + ".png" );
            }
        }
    }
#endif
    return toObj( mesh, out, settings, firstVertId );
}

Expected<void> toObj( const Mesh & mesh, std::ostream & out, const SaveSettings & settings, int firstVertId )
{
    MR_TIMER;
    out << "# MeshInspector.com\n";
    if ( settings.uvMap )
        out << fmt::format( "mtllib {}.mtl\n", settings.materialName );

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = mesh.topology.lastValidVert();

    int numSaved = 0;
    auto sb = subprogress( settings.progress, 0.0f, settings.uvMap ? 0.35f : 0.5f );
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !mesh.topology.hasVert( i ) )
            continue;

        auto saveVertex = [&]( auto && p )
        {
            if ( settings.colors )
            {
                const auto c = (Vector4f)( *settings.colors )[i];
                out << fmt::format( "v {} {} {} {} {} {}\n", p.x, p.y, p.z, c[0], c[1], c[2] );
            }
            else
            {
                out << fmt::format( "v {} {} {}\n", p.x, p.y, p.z );
            }
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, mesh.points[i] ) );
        else
            saveVertex( mesh.points[i] );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !sb( float( numSaved ) / numPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( settings.uvMap )
    {
        numSaved = 0;
        sb = subprogress( settings.progress, 0.35f, 0.7f );
        for ( VertId i{ 0 }; i <= lastVertId; ++i )
        {
            if ( settings.onlyValidPoints && !mesh.topology.hasVert( i ) )
                continue;
            const auto& uv = ( *settings.uvMap )[i];
            out << fmt::format( "vt {} {}\n", uv.x, uv.y );
            ++numSaved;
            if ( settings.progress && !( numSaved & 0x3FF ) && !sb( float( numSaved ) / numPoints ) )
                return unexpectedOperationCanceled();
        }
        out << "usemtl Texture\n";
    }

    sb = subprogress( settings.progress, settings.uvMap ? 0.7f : 0.5f, 1.0f );
    const float facesNum = float( mesh.topology.edgePerFace().size() );
    size_t faceIndex = 0;
    for ( const auto& e : mesh.topology.edgePerFace() )
    {
        ++faceIndex;
        if ( settings.progress && !( faceIndex & 0x3FF ) && !sb( faceIndex / facesNum ) )
            return unexpectedOperationCanceled();
        if ( !e.valid() )
            continue;

        VertId a, b, c;
        mesh.topology.getLeftTriVerts( e, a, b, c );
        Vector3i values( vertRenumber( a ) + firstVertId, vertRenumber( b ) + firstVertId, vertRenumber( c ) + firstVertId );
        if ( settings.uvMap )
            out << fmt::format( "f {}/{} {}/{} {}/{}\n",
                values.x, values.x,
                values.y, values.y,
                values.z, values.z );
        else
            out << fmt::format( "f {} {} {}\n",
                values.x, values.y, values.z );
    }

    if ( !out )
        return unexpected( std::string( "Error saving in OBJ-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toObj( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings& settings )
{
    return toObj( mesh, file, settings, 1 );
}

Expected<void> toObj( const Mesh& mesh, std::ostream& out, const SaveSettings& settings )
{
    return toObj( mesh, out, settings, 1 );
}

static FaceBitSet getNotDegenTris( const Mesh &mesh )
{
    MR_TIMER;
    FaceBitSet notDegenTris = mesh.topology.getValidFaces();
    BitSetParallelFor( notDegenTris, [&]( FaceId f )
    {
        VertId a, b, c;
        mesh.topology.getTriVerts( f, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );

        const Vector3f& ap = mesh.points[a];
        const Vector3f& bp = mesh.points[b];
        const Vector3f& cp = mesh.points[c];
        if ( ap == bp || bp == cp || cp == ap )
            notDegenTris.reset( f );
    } );
    return notDegenTris;
}

Expected<void> toBinaryStl( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toBinaryStl( mesh, out, settings );
}

Expected<void> toBinaryStl( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    char header[80] = "MeshInspector.com";
    out.write( header, 80 );

    auto notDegenTris = getNotDegenTris( mesh );
    auto numTris = (std::uint32_t)notDegenTris.count();
    out.write( ( const char* )&numTris, 4 );

    const float trisNum = float( notDegenTris.count() );
    int trisIndex = 0;
    for ( auto f : notDegenTris )
    {
        VertId a, b, c;
        mesh.topology.getTriVerts( f, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );

        // perform normal computation in double-precision to get exactly the same single-precision result on all platforms
        const Vector3d ad = applyDouble( settings.xf, mesh.points[a] );
        const Vector3d bd = applyDouble( settings.xf, mesh.points[b] );
        const Vector3d cd = applyDouble( settings.xf, mesh.points[c] );
        const Vector3f normal( cross( bd - ad, cd - ad ).normalized() );
        const Vector3f ap( ad );
        const Vector3f bp( bd );
        const Vector3f cp( cd );

        out.write( (const char*)&normal, 12 );
        out.write( (const char*)&ap, 12 );
        out.write( (const char*)&bp, 12 );
        out.write( (const char*)&cp, 12 );
        std::uint16_t attr{ 0 };
        out.write( ( const char* )&attr, 2 );
        if ( settings.progress && !( trisIndex & 0x3FF ) && !settings.progress( trisIndex / trisNum ) )
            return unexpectedOperationCanceled();
        ++trisIndex;
    }

    if ( !out )
        return unexpected( std::string( "Error saving in binary STL-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toAsciiStl( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toAsciiStl( mesh, out, settings );
}

Expected<void> toAsciiStl( const Mesh& mesh, std::ostream& out, const SaveSettings & settings )
{
    MR_TIMER;

    static const char* solid_name = "MeshInspector.com";
    out << "solid " << solid_name << "\n";
    auto notDegenTris = getNotDegenTris( mesh );
    const float trisNum = float( notDegenTris.count() );
    int trisIndex = 0;
    for ( auto f : notDegenTris )
    {
        VertId a, b, c;
        mesh.topology.getTriVerts( f, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );
        auto saveVertex = [&]( auto && ap, auto && bp, auto && cp )
        {
            const auto normal = cross( bp - ap, cp - ap ).normalized();
            out << "" << fmt::format( "facet normal {} {} {}\n", normal.x, normal.y, normal.z );
            out << "outer loop\n";
            for ( const auto & p : { ap, bp, cp } )
                out << fmt::format( "vertex {} {} {}\n", p.x, p.y, p.z );
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, mesh.points[a] ),
                        applyDouble( settings.xf, mesh.points[b] ),
                        applyDouble( settings.xf, mesh.points[c] ) );
        else
            saveVertex( mesh.points[a], mesh.points[b], mesh.points[c] );
        out << "endloop\n";
        out << "endfacet\n";
        if ( settings.progress && !( trisIndex & 0x3FF ) && !settings.progress( trisIndex / trisNum ) )
            return unexpectedOperationCanceled();
        ++trisIndex;
    }
    out << "endsolid " << solid_name << "\n";

    if ( !out )
        return unexpected( std::string( "Error saving in ascii STL-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toPly( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( mesh, out, settings );
}

Expected<void> toPly( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
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

Expected<void> toAnySupportedFormat( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto saver = getMeshSaver( ext );
    if ( !saver.fileSave )
        return unexpectedUnsupportedFileExtension();

    return saver.fileSave( mesh, file, settings );
}

Expected<void> toAnySupportedFormat( const Mesh& mesh, const std::string& extension, std::ostream& out, const SaveSettings & settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto saver = getMeshSaver( ext );
    if ( !saver.streamSave )
        return unexpected( std::string( "unsupported stream extension" ) );

    return saver.streamSave( mesh, out, settings );
}

MR_ADD_MESH_SAVER_WITH_PRIORITY( IOFilter( "MrMesh (.mrmesh)", "*.mrmesh" ), toMrmesh, {}, -1 )
MR_ADD_MESH_SAVER( IOFilter( "Binary STL (.stl)", "*.stl"   ), toBinaryStl, {} )
MR_ADD_MESH_SAVER( IOFilter( "OFF (.off)",        "*.off"   ), toOff, {} )
MR_ADD_MESH_SAVER( IOFilter( "OBJ (.obj)",        "*.obj"   ), toObj, { .storesVertexColors = true } )
MR_ADD_MESH_SAVER( IOFilter( "PLY (.ply)",        "*.ply"   ), toPly, { .storesVertexColors = true } )

} //namespace MeshSave

} //namespace MR
