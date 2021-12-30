#include "MRMeshSave.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "OpenCTM/openctm.h"

namespace MR
{

namespace MeshSave
{

const IOFilters Filters =
{
    {"MrMesh (.mrmesh)",  "*.mrmesh"},
    {"Binary STL (.stl)", "*.stl"},
    {"OFF (.off)",        "*.off"},
    {"OBJ (.obj)",        "*.obj"},
    {"PLY (.ply)",        "*.ply"},
    {"CTM (.ctm)",        "*.ctm"}
};

tl::expected<void, std::string> toMrmesh( const Mesh & mesh, const std::filesystem::path & file )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrmesh( mesh, out );
}

tl::expected<void, std::string> toMrmesh( const Mesh & mesh, std::ostream & out )
{
    MR_TIMER
    mesh.topology.write( out );

    // write points
    auto numPoints = (std::uint32_t)mesh.points.size();
    out.write( (const char*)&numPoints, 4 );
    out.write( (const char*)mesh.points.data(), mesh.points.size() * sizeof(Vector3f) );

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in Mrmesh-format" ) );

    return {};
}

tl::expected<void, std::string> toOff( const Mesh & mesh, const std::filesystem::path & file )
{
    std::ofstream out( file );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toOff( mesh, out );
}

tl::expected<void, std::string> toOff( const Mesh & mesh, std::ostream & out )
{
    MR_TIMER
    VertId lastValidPoint = mesh.topology.lastValidVert();
    int numPolygons = mesh.topology.numValidFaces();

    out << "OFF\n" << lastValidPoint+1 << ' ' << numPolygons << " 0\n\n";
    
    for ( VertId i{ 0 }; i <= lastValidPoint; ++i )
    {
        auto p = mesh.points[i];
        out << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    out << '\n';

    for ( const auto & e : mesh.topology.edgePerFace() )
    {
        if ( !e.valid() )
            continue;
        VertId a, b, c;
        mesh.topology.getLeftTriVerts( e, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );
        out << "3 " << a << ' ' << b << ' ' << c << '\n';
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in OFF-format" ) );

    return {};
}


tl::expected<void, std::string> toObj( const Mesh & mesh, const std::filesystem::path & file )
{
    std::ofstream out( file );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toObj( mesh, out );
}

tl::expected<void, std::string> toObj( const Mesh & mesh, std::ostream & out )
{
    MR_TIMER
    VertId lastValidPoint = mesh.topology.lastValidVert();

    for ( VertId i{ 0 }; i <= lastValidPoint; ++i )
    {
        auto p = mesh.points[i];
        out << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }

    for ( const auto & e : mesh.topology.edgePerFace() )
    {
        if ( !e.valid() )
            continue;
        VertId a, b, c;
        mesh.topology.getLeftTriVerts( e, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );
        out << "f " << a+1 << ' ' << b+1 << ' ' << c+1 << '\n';
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in OBJ-format" ) );

    return {};
}

tl::expected<void, std::string> toBinaryStl( const Mesh & mesh, const std::filesystem::path & file )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toBinaryStl( mesh, out );
}

tl::expected<void, std::string> toBinaryStl( const Mesh & mesh, std::ostream & out )
{
    MR_TIMER

    char header[80] = "MeshRUs";
    out.write( header, 80 );

    auto notDegenTris = mesh.topology.getValidFaces();
    for ( auto f : notDegenTris )
    {
        VertId a, b, c;
        mesh.topology.getTriVerts( f, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );

        const Vector3f & ap = mesh.points[a];
        const Vector3f & bp = mesh.points[b];
        const Vector3f & cp = mesh.points[c];
        if ( ap == bp || bp == cp || cp == ap )
            notDegenTris.reset( f );
    }

    auto numTris = (std::uint32_t)notDegenTris.count();
    out.write( (const char*)&numTris, 4 );

    for ( auto f : notDegenTris )
    {
        VertId a, b, c;
        mesh.topology.getTriVerts( f, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );

        const Vector3f & ap = mesh.points[a];
        const Vector3f & bp = mesh.points[b];
        const Vector3f & cp = mesh.points[c];
        Vector3f normal = cross( bp - ap, cp - ap ).normalized();
        out.write( (const char*)&normal, 12 );
        out.write( (const char*)&ap, 12 );
        out.write( (const char*)&bp, 12 );
        out.write( (const char*)&cp, 12 );
        std::uint16_t attr{ 0 };
        out.write( (const char*)&attr, 2 );
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in binary STL-format" ) );

    return {};
}

tl::expected<void, std::string> toPly( const Mesh & mesh, const std::filesystem::path & file, const std::vector<Color>* perVertColors )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( mesh, out, perVertColors );
}

tl::expected<void, std::string> toPly( const Mesh & mesh, std::ostream & out, const std::vector<Color>* perVertColors )
{
    MR_TIMER

    int numVertices = mesh.topology.lastValidVert() + 1;
    bool saveColors = perVertColors && perVertColors->size() >= numVertices;

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshRUs\n"
        "element vertex " << numVertices << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( saveColors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";

    out <<  "element face " << mesh.topology.numValidFaces() << "\nproperty list uchar int vertex_indices\nend_header\n";

    // write vertices
    static_assert( sizeof( mesh.points.front() ) == 12, "wrong size of Vector3f" );
    if ( !saveColors )
    {
        out.write( (const char*) &mesh.points.front().x, 12 * numVertices );
    }
    else
    {
        static_assert( sizeof( perVertColors->front() ) == 4, "wrong size of Color" );
        for ( int i = 0; i < numVertices; ++i )
        {
            out.write( (const char*) &mesh.points[VertId( i )].x, 12 );
            out.write( (const char*) &( *perVertColors )[i].r, 3 ); // write only r g b, not a
        }
    }

    // write triangles
    #pragma pack(push, 1)
    struct PlyTriangle
    {
        char cnt = 3;
        VertId v[3];
    };
    #pragma pack(pop)
    static_assert( sizeof( PlyTriangle ) == 13, "check your padding" );

    PlyTriangle tri;
    for ( auto f : mesh.topology.getValidFaces() )
    {
        mesh.topology.getTriVerts( f, tri.v );
        out.write( (const char *)&tri, 13 );
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in PLY-format" ) );

    return {};
}

tl::expected<void, std::string> toCtm( const Mesh & mesh, const std::filesystem::path & file, const CtmSaveOptions options )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( mesh, out, options );
}

tl::expected<void, std::string> toCtm( const Mesh & mesh, std::ostream & out, const CtmSaveOptions options )
{
    MR_TIMER

    class ScopedCtmConext 
    {
        CTMcontext context_ = ctmNewContext( CTM_EXPORT );
    public:
        ~ScopedCtmConext() { ctmFreeContext( context_ ); }
        operator CTMcontext() { return context_; }
    } context;

    ctmFileComment( context, options.comment );
    switch ( options.meshCompression )
    {
    default:
        assert( false );
        [[fallthrough]];
    case CtmSaveOptions::MeshCompression::None:
        ctmCompressionMethod( context, CTM_METHOD_RAW );
        break;
    case CtmSaveOptions::MeshCompression::Lossless:
        ctmCompressionMethod( context, CTM_METHOD_MG1 );
        break;
    case CtmSaveOptions::MeshCompression::Lossy:
        ctmCompressionMethod( context, CTM_METHOD_MG2 );
        ctmVertexPrecision( context, options.vertexPrecision );
        break;
    }
    ctmRearrangeTriangles( context, options.rearrangeTriangles ? 1 : 0 );
    ctmCompressionLevel( context, options.compressionLevel );

    std::vector<CTMuint> aIndices;
    const auto fLast = mesh.topology.lastValidFace();
    const auto numSaveFaces = options.rearrangeTriangles ? mesh.topology.numValidFaces()  : int( fLast + 1 );
    aIndices.reserve( numSaveFaces * 3 );
    for ( FaceId f{0}; f <= fLast; ++f )
    {
        VertId v[3];
        if ( mesh.topology.hasFace( f ) )
            mesh.topology.getTriVerts( f, v );
        else if( options.rearrangeTriangles )
            continue;
        else
            v[0] = v[1] = v[2] = 0_v;
        aIndices.push_back( v[0] );
        aIndices.push_back( v[1] );
        aIndices.push_back( v[2] );
    }
    assert( aIndices.size() == numSaveFaces * 3 );

    CTMuint aVertexCount = mesh.topology.lastValidVert() + 1;
    ctmDefineMesh( context, 
        (const CTMfloat *)mesh.points.data(), aVertexCount, 
        aIndices.data(), numSaveFaces, nullptr );

    if ( ctmGetError(context) != CTM_NONE )
        return tl::make_unexpected( "Error encoding in CTM-format" );

    ctmSaveCustom( context, []( const void * buf, CTMuint size, void * data )
    {
        std::ostream & s = *reinterpret_cast<std::ostream *>( data );
        s.write( (const char*)buf, size );
        return s.good() ? size : 0;
    }, &out );

    if ( !out || ctmGetError(context) != CTM_NONE )
        return tl::make_unexpected( std::string( "Error saving in CTM-format" ) );

    return {};
}

tl::expected<void, std::string> toAnySupportedFormat( const Mesh& mesh, const std::filesystem::path& file, const std::vector<Color>* perVertColors )
{
    auto ext = file.extension().u8string();
    for ( auto & c : ext )
        c = (char) tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".off" )
        res = MR::MeshSave::toOff( mesh, file );
    else if ( ext == u8".obj" )
        res = MR::MeshSave::toObj( mesh, file );
    else if ( ext == u8".stl" )
        res = MR::MeshSave::toBinaryStl( mesh, file );
    else if ( ext == u8".ply" )
        res = MR::MeshSave::toPly( mesh, file, perVertColors );
    else if ( ext == u8".ctm" )
        res = MR::MeshSave::toCtm( mesh, file );
    else if ( ext == u8".mrmesh" )
        res = MR::MeshSave::toMrmesh( mesh, file );
    return res;
}

} //namespace MeshSave

} //namespace MR
