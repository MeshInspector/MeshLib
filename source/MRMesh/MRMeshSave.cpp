#include "MRMeshSave.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRBitSetParallelFor.h"
#include "MRPch/MRFmt.h"

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

namespace MR
{

namespace MeshSave
{

namespace
{

class VertRenumber
{
public:
    VertRenumber( const VertBitSet & validVerts, bool saveValidOnly )
    {
        MR_TIMER
        if ( saveValidOnly )
        {
            vert2packed_ = makeVectorWithSeqNums( validVerts );
            sizeVerts_ = (int)validVerts.count();
        }
        else
            sizeVerts_ = validVerts.find_last() + 1;
    }
    int operator()( VertId v ) const
    {
        assert( v );
        return vert2packed_.empty() ? (int)v : vert2packed_[v];
    }
    int sizeVerts() const
    {
        return sizeVerts_;
    }
private:
    Vector<int, VertId> vert2packed_;
    int sizeVerts_ = 0;
};

} //anonymous namespace

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

VoidOrErrStr toMrmesh( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrmesh( mesh, out, settings );
}

VoidOrErrStr toMrmesh( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER
    mesh.topology.write( out );

    // write points
    auto numPoints = (std::uint32_t)( mesh.topology.lastValidVert() + 1 );
    out.write( (const char*)&numPoints, 4 );

    VertCoords buf;
    const auto & xfVerts = transformPoints( mesh.points, mesh.topology.getValidVerts(), settings.xf, buf );
    if ( !writeByBlocks( out, ( const char* )xfVerts.data(), numPoints * sizeof( Vector3f ), settings.progress ) )
        return unexpected( std::string( "Saving canceled" ) );

    if ( !out )
        return unexpected( std::string( "Error saving in Mrmesh-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

VoidOrErrStr toOff( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toOff( mesh, out, settings );
}

VoidOrErrStr toOff( const Mesh& mesh, std::ostream& out, const SaveSettings & settings )
{
    MR_TIMER

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), settings.saveValidOnly );
    const int maxPoints = vertRenumber.sizeVerts();
    const int numPolygons = mesh.topology.numValidFaces();

    out << "OFF\n" << maxPoints << ' ' << numPolygons << " 0\n\n";
    int numSaved = 0;
    for ( VertId i{ 0 }; i < maxPoints; ++i )
    {
        if ( settings.saveValidOnly && !mesh.topology.hasVert( i ) )
            continue;
        auto p = applyDouble( settings.xf, mesh.points[i] );
        out << fmt::format( "{} {} {}\n", p.x, p.y, p.z );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / maxPoints * 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
    }
    out << '\n';

    const float facesNum = float( mesh.topology.edgePerFace().size() );
    size_t faceIndex = 0;
    for ( const auto& e : mesh.topology.edgePerFace() )
    {
        ++faceIndex;
        if ( settings.progress && !( faceIndex & 0x3FF ) && !settings.progress( float( faceIndex ) / facesNum * 0.5f + 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
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


VoidOrErrStr toObj( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, int firstVertId )
{
    std::ofstream out( file );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toObj( mesh, out, settings, firstVertId );
}

VoidOrErrStr toObj( const Mesh & mesh, std::ostream & out, const SaveSettings & settings, int firstVertId )
{
    MR_TIMER
    out << "# MeshInspector.com\n";

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), settings.saveValidOnly );
    const int maxPoints = vertRenumber.sizeVerts();

    int numSaved = 0;
    for ( VertId i{ 0 }; i < maxPoints; ++i )
    {
        if ( settings.saveValidOnly && !mesh.topology.hasVert( i ) )
            continue;
        auto p = applyDouble( settings.xf, mesh.points[i] );
        out << fmt::format( "v {} {} {}\n", p.x, p.y, p.z );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / maxPoints * 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
    }

    const float facesNum = float( mesh.topology.edgePerFace().size() );
    size_t faceIndex = 0;
    for ( const auto& e : mesh.topology.edgePerFace() )
    {
        ++faceIndex;
        if ( settings.progress && !( faceIndex & 0x3FF ) && !settings.progress( faceIndex / facesNum * 0.5f + 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
        if ( !e.valid() )
            continue;

        VertId a, b, c;
        mesh.topology.getLeftTriVerts( e, a, b, c );
        out << fmt::format( "f {} {} {}\n",
            vertRenumber( a ) + firstVertId,
            vertRenumber( b ) + firstVertId,
            vertRenumber( c ) + firstVertId );
    }

    if ( !out )
        return unexpected( std::string( "Error saving in OBJ-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

static FaceBitSet getNotDegenTris( const Mesh &mesh )
{
    MR_TIMER
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

VoidOrErrStr toBinaryStl( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toBinaryStl( mesh, out, settings );
}

VoidOrErrStr toBinaryStl( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
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

        const Vector3f ap = applyFloat( settings.xf, mesh.points[a] );
        const Vector3f bp = applyFloat( settings.xf, mesh.points[b] );
        const Vector3f cp = applyFloat( settings.xf, mesh.points[c] );
        Vector3f normal = cross( bp - ap, cp - ap ).normalized();
        out.write( (const char*)&normal, 12 );
        out.write( (const char*)&ap, 12 );
        out.write( (const char*)&bp, 12 );
        out.write( (const char*)&cp, 12 );
        std::uint16_t attr{ 0 };
        out.write( ( const char* )&attr, 2 );
        if ( settings.progress && !( trisIndex & 0x3FF ) && !settings.progress( trisIndex / trisNum ) )
            return unexpected( std::string( "Saving canceled" ) );
        ++trisIndex;
    }

    if ( !out )
        return unexpected( std::string( "Error saving in binary STL-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

VoidOrErrStr toAsciiStl( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toAsciiStl( mesh, out, settings );
}

VoidOrErrStr toAsciiStl( const Mesh& mesh, std::ostream& out, const SaveSettings & settings )
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
        const auto ap = applyDouble( settings.xf, mesh.points[a] );
        const auto bp = applyDouble( settings.xf, mesh.points[b] );
        const auto cp = applyDouble( settings.xf, mesh.points[c] );
        const auto normal = cross( bp - ap, cp - ap ).normalized();
        out << "" << fmt::format( "facet normal {} {} {}\n", normal.x, normal.y, normal.z );
        out << "outer loop\n";
        for ( const auto & p : { ap, bp, cp } )
        {
            out << fmt::format( "vertex {} {} {}\n", p.x, p.y, p.z );
        }
        out << "endloop\n";
        out << "endfacet\n";
        if ( settings.progress && !( trisIndex & 0x3FF ) && !settings.progress( trisIndex / trisNum ) )
            return unexpected( std::string( "Saving canceled" ) );
        ++trisIndex;
    }
    out << "endsolid " << solid_name << "\n";

    if ( !out )
        return unexpected( std::string( "Error saving in ascii STL-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

VoidOrErrStr toPly( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( mesh, out, settings );
}

VoidOrErrStr toPly( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER

    int numVertices = mesh.topology.lastValidVert() + 1;
    bool saveColors = settings.colors && settings.colors->size() >= numVertices;

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n"
        "element vertex " << numVertices << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( saveColors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";

    out <<  "element face " << mesh.topology.numValidFaces() << "\nproperty list uchar int vertex_indices\nend_header\n";

    // write vertices
    static_assert( sizeof( mesh.points.front() ) == 12, "wrong size of Vector3f" );
    if ( !saveColors )
    {
        VertCoords buf;
        const auto & xfVerts = transformPoints( mesh.points, mesh.topology.getValidVerts(), settings.xf, buf );
        if ( !writeByBlocks( out, ( const char* )xfVerts.data(), numVertices * sizeof( Vector3f ), subprogress( settings.progress, 0.0f, 0.5f ) ) )
            return unexpected( std::string( "Saving canceled" ) );
    }
    else
    {
        static_assert( sizeof( settings.colors->front() ) == 4, "wrong size of Color" );
        for ( VertId i{ 0 }; i < numVertices; ++i )
        {
            const Vector3f p = applyFloat( settings.xf, mesh.points[i] );
            out.write( (const char*) &p.x, 12 );
            out.write( (const char*) &( *settings.colors )[i].r, 3 ); // write only r g b, not a
            if ( settings.progress && !( i & 0x3FF ) && !settings.progress( float( i ) / numVertices * 0.5f ) )
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
        if ( settings.progress && !( faceIndex & 0x3FF ) && !settings.progress( float( faceIndex ) / facesNum * 0.5f + 0.5f ) )
            return unexpected( std::string( "Saving canceled" ) );
        ++faceIndex;
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PLY-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

#ifndef MRMESH_NO_OPENCTM
VoidOrErrStr toCtm( const Mesh & mesh, const std::filesystem::path & file, const CtmSaveOptions options )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( mesh, out, options );
}

VoidOrErrStr toCtm( const Mesh & mesh, std::ostream & out, const CtmSaveOptions options )
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
    const auto numSaveFaces = options.rearrangeTriangles ? mesh.topology.numValidFaces() : int( fLast + 1 );
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
    VertCoords buf;
    const auto & xfVerts = transformPoints( mesh.points, mesh.topology.getValidVerts(), options.xf, buf );
    ctmDefineMesh( context,
        (const CTMfloat *)xfVerts.data(), aVertexCount, 
        aIndices.data(), numSaveFaces, nullptr );

    if ( ctmGetError(context) != CTM_NONE )
        return unexpected( "Error encoding in CTM-format" );

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( options.colors )
    {
        colors4f.resize( aVertexCount );
        const auto maxV = (int)std::min( aVertexCount, (CTMuint)options.colors->size() );
        for ( VertId i{ 0 }; i < maxV; ++i )
            colors4f[i] = Vector4f( ( *options.colors )[i] );

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
    if ( options.progress )
    {
        if ( options.meshCompression == CtmSaveOptions::MeshCompression::None )
        {
            saveData.callbackFn = [callback = options.progress, &saveData] ( float progress )
            {
                // calculate full progress
                progress = ( saveData.sum + progress * saveData.blockSize ) / saveData.maxSize;
                return callback( progress );
            };
        }
        else
        {
            saveData.callbackFn = [callback = options.progress, &saveData] ( float progress )
            {
                // calculate full progress in partial-linear scale (we don't know compressed size and it less than real size)
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

    reportProgress( options.progress, 1.f );
    return {};
}
#endif

VoidOrErrStr toAnySupportedFormat( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char) tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".off" )
        res = MR::MeshSave::toOff( mesh, file, settings );
    else if ( ext == ".obj" )
        res = MR::MeshSave::toObj( mesh, file, settings );
    else if ( ext == ".stl" )
        res = MR::MeshSave::toBinaryStl( mesh, file, settings );
    else if ( ext == ".ply" )
        res = MR::MeshSave::toPly( mesh, file, settings );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
    {
        CtmSaveOptions options{ settings };
        options.rearrangeTriangles = true;
        res = MR::MeshSave::toCtm( mesh, file, options );
    }
#endif
    else if ( ext == ".mrmesh" )
        res = MR::MeshSave::toMrmesh( mesh, file, settings );
    return res;
}

VoidOrErrStr toAnySupportedFormat( const Mesh& mesh, std::ostream& out, const std::string& extension, const SaveSettings & settings )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".off" )
        res = MR::MeshSave::toOff( mesh, out, settings );
    else if ( ext == ".obj" )
        res = MR::MeshSave::toObj( mesh, out, settings );
    else if ( ext == ".stl" )
        res = MR::MeshSave::toBinaryStl( mesh, out, settings );
    else if ( ext == ".ply" )
        res = MR::MeshSave::toPly( mesh, out, settings );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
    {
        CtmSaveOptions options{ settings };
        options.rearrangeTriangles = true;
        res = MR::MeshSave::toCtm( mesh, out, options );
    }
#endif
    else if ( ext == ".mrmesh" )
        res = MR::MeshSave::toMrmesh( mesh, out, settings );
    return res;
}

} //namespace MeshSave

} //namespace MR
