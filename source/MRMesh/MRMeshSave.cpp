#include "MRMeshSave.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRPch/MRSpdlog.h"

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

#if _MSC_VER <= 1929 // Visual Studio 2019
#pragma warning( disable : 4866 ) //compiler may not enforce left-to-right evaluation order for call to 'std::operator<<<char,std::char_traits<char>,std::allocator<char> >'
#endif

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
#ifndef MRMESH_NO_OPENCTM
    {"CTM (.ctm)",        "*.ctm"},
#endif
};

VoidOrErrStr toMrmesh( const Mesh & mesh, const std::filesystem::path & file, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrmesh( mesh, out, callback );
}

VoidOrErrStr toMrmesh( const Mesh & mesh, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER
    mesh.topology.write( out );

    // write points
    auto numPoints = (std::uint32_t)mesh.points.size();
    out.write( (const char*)&numPoints, 4 );

    const bool cancel = !MR::writeByBlocks( out, ( const char* )mesh.points.data(), mesh.points.size() * sizeof( Vector3f ), callback );
    if ( cancel )
        return unexpected( std::string( "Saving canceled" ) );

    if ( !out )
        return unexpected( std::string( "Error saving in Mrmesh-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

VoidOrErrStr toOff( const Mesh & mesh, const std::filesystem::path & file, ProgressCallback callback )
{
    std::ofstream out( file );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toOff( mesh, out, callback );
}

VoidOrErrStr toOff( const Mesh& mesh, std::ostream& out, ProgressCallback callback )
{
    MR_TIMER
        VertId maxPoints = mesh.topology.lastValidVert() + 1;
    int numPolygons = mesh.topology.numValidFaces();

    out << "OFF\n" << maxPoints << ' ' << numPolygons << " 0\n\n";
    for ( VertId i{ 0 }; i < maxPoints; ++i )
    {
        auto p = mesh.points[i];
        out << fmt::format( "{} {} {}\n", p.x, p.y, p.z );
        if ( callback && !( i & 0x3FF ) && !callback( float( i ) / maxPoints * 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
    }
    out << '\n';

    const float facesNum = float( mesh.topology.edgePerFace().size() );
    size_t faceIndex = 0;
    for ( const auto& e : mesh.topology.edgePerFace() )
    {
        ++faceIndex;
        if ( callback && !( faceIndex & 0x3FF ) && !callback( float( faceIndex ) / facesNum * 0.5f + 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
        if ( !e.valid() )
            continue;

        VertId a, b, c;
        mesh.topology.getLeftTriVerts( e, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );
        out << fmt::format( "3 {} {} {}\n", (int)a, (int)b, (int)c );
    }

    if ( !out )
        return unexpected( std::string( "Error saving in OFF-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}


VoidOrErrStr toObj( const Mesh & mesh, const std::filesystem::path & file, const AffineXf3f & xf, int firstVertId,
                                       ProgressCallback callback )
{
    std::ofstream out( file );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toObj( mesh, out, xf, firstVertId, callback );
}

VoidOrErrStr toObj( const Mesh & mesh, std::ostream & out, const AffineXf3f & xf, int firstVertId,
                                       ProgressCallback callback )
{
    MR_TIMER
    out << "# MeshInspector.com\n";

    VertId lastValidPoint = mesh.topology.lastValidVert();

    for ( VertId i{ 0 }; i <= lastValidPoint; ++i )
    {
        auto p = xf( mesh.points[i] );
        out << fmt::format( "v {} {} {}\n", p.x, p.y, p.z );
        if ( callback && !( i & 0x3FF ) && !callback( float( i ) / lastValidPoint * 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
    }

    const float facesNum = float( mesh.topology.edgePerFace().size() );
    size_t faceIndex = 0;
    for ( const auto& e : mesh.topology.edgePerFace() )
    {
        ++faceIndex;
        if ( callback && !( faceIndex & 0x3FF ) && !callback( faceIndex / facesNum * 0.5f + 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
        if ( !e.valid() )
            continue;

        VertId a, b, c;
        mesh.topology.getLeftTriVerts( e, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );
        out << fmt::format( "f {} {} {}\n", int( a + firstVertId ), int( b + firstVertId ), int( c + firstVertId ) );
    }

    if ( !out )
        return unexpected( std::string( "Error saving in OBJ-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

static FaceBitSet getNotDegenTris( const Mesh &mesh )
{
    FaceBitSet notDegenTris = mesh.topology.getValidFaces();
    VertId a, b, c;
    for ( auto f : notDegenTris )
    {
        mesh.topology.getTriVerts( f, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );

        const Vector3f& ap = mesh.points[a];
        const Vector3f& bp = mesh.points[b];
        const Vector3f& cp = mesh.points[c];
        if ( ap == bp || bp == cp || cp == ap )
            notDegenTris.reset( f );
    }
    return notDegenTris;
}

VoidOrErrStr toBinaryStl( const Mesh & mesh, const std::filesystem::path & file, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toBinaryStl( mesh, out, callback );
}

VoidOrErrStr toBinaryStl( const Mesh & mesh, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER

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

        const Vector3f& ap = mesh.points[a];
        const Vector3f& bp = mesh.points[b];
        const Vector3f& cp = mesh.points[c];
        Vector3f normal = cross( bp - ap, cp - ap ).normalized();
        out.write( (const char*)&normal, 12 );
        out.write( (const char*)&ap, 12 );
        out.write( (const char*)&bp, 12 );
        out.write( (const char*)&cp, 12 );
        std::uint16_t attr{ 0 };
        out.write( ( const char* )&attr, 2 );
        if ( callback && !( trisIndex & 0x3FF ) && !callback( trisIndex / trisNum ) )
            return unexpected( std::string( "Saving canceled" ) );
        ++trisIndex;
    }

    if ( !out )
        return unexpected( std::string( "Error saving in binary STL-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

VoidOrErrStr toAsciiStl( const Mesh& mesh, const std::filesystem::path& file, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toAsciiStl( mesh, out, callback );
}

VoidOrErrStr toAsciiStl( const Mesh& mesh, std::ostream& out, ProgressCallback callback )
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
        const Vector3f& ap = mesh.points[a];
        const Vector3f& bp = mesh.points[b];
        const Vector3f& cp = mesh.points[c];
        Vector3f normal = cross( bp - ap, cp - ap ).normalized();
        out << "" << fmt::format( "facet normal {} {} {}\n", normal.x, normal.y, normal.z );
        out << "outer loop\n";
        for ( const Vector3f& p : { ap, bp, cp } )
        {
            out << fmt::format( "vertex {} {} {}\n", p.x, p.y, p.z );
        }
        out << "endloop\n";
        out << "endfacet\n";
        if ( callback && !( trisIndex & 0x3FF ) && !callback( trisIndex / trisNum ) )
            return unexpected( std::string( "Saving canceled" ) );
        ++trisIndex;
    }
    out << "endsolid " << solid_name << "\n";

    if ( !out )
        return unexpected( std::string( "Error saving in ascii STL-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

VoidOrErrStr toPly( const Mesh & mesh, const std::filesystem::path & file, const VertColors* colors, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( mesh, out, colors, callback );
}

VoidOrErrStr toPly( const Mesh & mesh, std::ostream & out, const VertColors* colors, ProgressCallback callback )
{
    MR_TIMER

    int numVertices = mesh.topology.lastValidVert() + 1;
    bool saveColors = colors && colors->size() >= numVertices;

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n"
        "element vertex " << numVertices << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( saveColors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";

    out <<  "element face " << mesh.topology.numValidFaces() << "\nproperty list uchar int vertex_indices\nend_header\n";

    // write vertices
    static_assert( sizeof( mesh.points.front() ) == 12, "wrong size of Vector3f" );
    if ( !saveColors )
    {
        ProgressCallback callbackFn = {};
        if (callback)
            callbackFn = [callback] ( float v )
            {
                return callback( v / 2.f );
            };
        const bool cancel = !MR::writeByBlocks( out, ( const char* )mesh.points.data(), numVertices * sizeof( Vector3f ), callbackFn );
        if ( cancel )
            return unexpected( std::string( "Saving canceled" ) );
    }
    else
    {
        static_assert( sizeof( colors->front() ) == 4, "wrong size of Color" );
        for ( VertId i{ 0 }; i < numVertices; ++i )
        {
            out.write( (const char*) &mesh.points[i].x, 12 );
            out.write( (const char*) &( *colors )[i].r, 3 ); // write only r g b, not a
            if ( callback && !( i & 0x3FF ) && !callback( float( i ) / numVertices * 0.5f ) )
                return unexpected( std::string( "Saving canceled" ) );
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
    const float facesNum = float( mesh.topology.getValidFaces().count() );
    int faceIndex = 0;
    for ( auto f : mesh.topology.getValidFaces() )
    {
        mesh.topology.getTriVerts( f, tri.v );
        out.write( (const char *)&tri, 13 );
        if ( callback && !( faceIndex & 0x3FF ) && !callback( float( faceIndex ) / facesNum * 0.5f + 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
        ++faceIndex;
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PLY-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

#ifndef MRMESH_NO_OPENCTM
VoidOrErrStr toCtm( const Mesh & mesh, const std::filesystem::path & file, const CtmSaveOptions options, const VertColors* colors,
                                       ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( mesh, out, options, colors, callback );
}

VoidOrErrStr toCtm( const Mesh & mesh, std::ostream & out, const CtmSaveOptions options, const VertColors* colors,
                                       ProgressCallback callback )
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
        return unexpected( "Error encoding in CTM-format" );

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( colors )
    {
        colors4f.resize( aVertexCount );
        const auto maxV = (int)std::min( aVertexCount, (CTMuint)colors->size() );
        for ( VertId i{ 0 }; i < maxV; ++i )
            colors4f[i] = Vector4f( ( *colors )[i] );

        ctmAddAttribMap( context, (const CTMfloat*) colors4f.data(), "Color" );
    }

    if ( ctmGetError( context ) != CTM_NONE )
        return unexpected( "Error encoding in CTM-format colors" );

    struct SaveData
    {
        std::function<bool( float )> callbackFn{};
        std::ostream* stream;
        size_t sum{ 0 };
        size_t blockSize{ 0 };
        size_t maxSize{ 0 };
        bool wasCanceled{ false };
    } saveData;
    if ( callback )
    {
        if ( options.meshCompression == CtmSaveOptions::MeshCompression::None )
        {
            saveData.callbackFn = [callback, &saveData] ( float progress )
            {
                // calculate full progress
                progress = ( saveData.sum + progress * saveData.blockSize ) / saveData.maxSize;
                return callback( progress );
            };
        }
        else
        {
            saveData.callbackFn = [callback, &saveData] ( float progress )
            {
                // calculate full progress in partical-linear scale (we don't know compressed size and it less than real size)
                // conversion rules:
                // step 1) range (0, rangeBefore) is converted in range (0, rangeAfter)
                // step 2) moving on to new ranges: (rangeBefore, 1) and (rangeAfter, 1)
                // step 3) go to step 1)
                const float rangeBefore = 0.2f;
                const float rangeAfter = 0.7f;
                progress = ( saveData.sum + progress * saveData.blockSize ) / saveData.maxSize;
                float newProgress = 0.f;
                for ( ; newProgress < 98.5f; )
                {
                    if ( progress < rangeBefore )
                    {
                        newProgress += progress / rangeBefore * rangeAfter * ( 1 - newProgress );
                        break;
                    }
                    else
                    {
                        progress = ( progress - rangeBefore ) / ( 1 - rangeBefore );
                        newProgress += ( 1 - newProgress ) * rangeAfter;
                    }
                }
                return callback( newProgress );
            };
        }
    }
    saveData.stream = &out;
    saveData.maxSize = mesh.points.size() * sizeof( Vector3f ) + mesh.topology.getValidFaces().count() * 3 * sizeof( int ) + 150; // 150 - reserve for some ctm specific data
    ctmSaveCustom( context, []( const void * buf, CTMuint size, void * data )
    {
        SaveData& saveData = *reinterpret_cast< SaveData* >( data );
        std::ostream& outStream = *saveData.stream;
        saveData.blockSize = size;

        saveData.wasCanceled |= !MR::writeByBlocks( outStream, (const char*) buf, size, saveData.callbackFn, 1u << 12 );
        saveData.sum += size;
        if ( saveData.wasCanceled )
            return 0u;

        return outStream.good() ? size : 0;
    }, &saveData );

    if ( saveData.wasCanceled )
        return unexpected( std::string( "Saving canceled" ) );
    if ( !out || ctmGetError(context) != CTM_NONE )
        return unexpected( std::string( "Error saving in CTM-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}
#endif

VoidOrErrStr toAnySupportedFormat( const Mesh& mesh, const std::filesystem::path& file, const VertColors* colors,
                                                      ProgressCallback callback )
{
    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char) tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".off" )
        res = MR::MeshSave::toOff( mesh, file, callback );
    else if ( ext == ".obj" )
        res = MR::MeshSave::toObj( mesh, file, {}, 1, callback );
    else if ( ext == ".stl" )
        res = MR::MeshSave::toBinaryStl( mesh, file, callback );
    else if ( ext == ".ply" )
        res = MR::MeshSave::toPly( mesh, file, colors, callback );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::MeshSave::toCtm( mesh, file, { .rearrangeTriangles = true }, colors, callback );
#endif
    else if ( ext == ".mrmesh" )
        res = MR::MeshSave::toMrmesh( mesh, file, callback );
    return res;
}

VoidOrErrStr toAnySupportedFormat( const Mesh& mesh, std::ostream& out, const std::string& extension,
                                                      const VertColors* colors /*= nullptr */, ProgressCallback callback )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".off" )
        res = MR::MeshSave::toOff( mesh, out, callback );
    else if ( ext == ".obj" )
        res = MR::MeshSave::toObj( mesh, out, {}, 1, callback );
    else if ( ext == ".stl" )
        res = MR::MeshSave::toBinaryStl( mesh, out, callback );
    else if ( ext == ".ply" )
        res = MR::MeshSave::toPly( mesh, out, colors, callback );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::MeshSave::toCtm( mesh, out, {}, colors, callback );
#endif
    else if ( ext == ".mrmesh" )
        res = MR::MeshSave::toMrmesh( mesh, out, callback );
    return res;
}

} //namespace MeshSave

} //namespace MR
