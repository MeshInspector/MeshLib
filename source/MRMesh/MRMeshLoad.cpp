#include "MRMeshLoad.h"
#include "MRMeshBuilder.h"
#include "MRIdentifyVertices.h"
#include "MRMesh.h"
#include "MRphmap.h"
#include "MRTimer.h"
#include "miniply.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"
#include "MRMeshLoadObj.h"
#include "MRMeshLoadStep.h"
#include "MRObjectMesh.h"
#include "MRObjectsAccess.h"
#include "MRColor.h"
#include "MRProgressReadWrite.h"
#include "MRIOParsing.h"
#include "MRMeshDelone.h"
#include "MRParallelFor.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRTBB.h"

#include <array>
#include <future>

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
#include <tinyxml2.h>
#endif

namespace MR
{

namespace MeshLoad
{

Expected<Mesh> fromMrmesh( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromMrmesh( in, settings ), file );
}

Expected<Mesh> fromMrmesh( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    Mesh mesh;
    auto readRes = mesh.topology.read( in, subprogress( settings.callback, 0.f, 0.5f) );
    if ( !readRes.has_value() )
    {
        std::string error = readRes.error();
        if ( error != "Loading canceled" )
            error = "Error reading topology from mrmesh - file:\n" + error;
        return unexpected( error );
    }

    // read points
    std::uint32_t numPoints;
    in.read( (char*)&numPoints, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the number of points from mrmesh-file" ) );
    mesh.points.resize( numPoints );
    if ( !readByBlocks( in, ( char* )mesh.points.data(), mesh.points.size() * sizeof( Vector3f ), subprogress( settings.callback, 0.5f, 1.f ) ) )
        return unexpected( std::string( "Loading canceled" ) );

    if ( !in )
        return unexpected( std::string( "Error reading  points from mrmesh-file" ) );

    return mesh;
}

Expected<Mesh> fromOff( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromOff( in, settings ), file );
}

Expected<Mesh> fromOff( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    auto bufOrExpect = readCharBuffer( in );

    if ( !bufOrExpect )
        return unexpected( std::move( bufOrExpect.error() ) );

    auto& buf = bufOrExpect.value();

    auto splitLines = splitByLines( buf.data(), buf.size() );
    in.seekg( 0);
    std::string header;
    in >> header;
    if ( !in || header != "OFF" )
        return unexpected( std::string( "File is not in OFF-format" ) );

    int numPoints, numPolygons, numUnused;
    in >> numPoints >> numPolygons >> numUnused;
    if ( !in || numPoints <= 0 || numPolygons <= 0 || numUnused != 0 )
        return unexpected( std::string( "Unsupported OFF-format" ) );

    size_t strHeader = 2;
    for ( size_t i = 2; i < splitLines.size(); i++ )
    {
        if ( splitLines[i + 1] - splitLines[i] > 2 )
        {
            strHeader = i;
            break;
        }
    }
    
    size_t strBorder = 0;

    for ( size_t i = numPoints + strHeader; i < splitLines.size(); i++ )
    {
        if ( splitLines[i + 1] - splitLines[i] > 2 )
        {
            strBorder = i - (numPoints + strHeader);
            break;
        }
    }

    std::vector<Vector3f> pointsBlocks( numPoints );
    
    std::atomic<bool> forseStop = false;
    bool keepGoing = ParallelFor( pointsBlocks, [&] ( size_t numPoint )
    {
        if ( forseStop )
        {
            return;
        }
        size_t numLine = strHeader + numPoint;

        const std::string_view line( &buf[splitLines[numLine]], splitLines[numLine + 1] - splitLines[numLine] );
        Vector3d temp;
        auto result = parseTextCoordinate( line, temp );
        pointsBlocks[numPoint] = Vector3f( temp );

        if ( !result.has_value() )
        {
            forseStop = true;
        }
    }, settings.callback );

    if ( forseStop )
    {
        return unexpected( std::string( "Error when reading coordinates" ) );
    }
    if ( !keepGoing )
    {
        return unexpectedOperationCanceled();
    }

    size_t delta = numPoints + strHeader + strBorder;

    Vector<MeshBuilder::VertSpan, FaceId> faces( numPolygons );
    int numPolygonPoint = 0;
    int start = 0;
    for ( size_t i = 0; i < numPolygons; i++ )
    {
        size_t numLine = delta + i;
        
        const std::string_view line( &buf[splitLines[numLine]], splitLines[numLine + 1] - splitLines[numLine] );
        parseFirstNum( line, numPolygonPoint );

        faces.vec_[i] = MeshBuilder::VertSpan{ start, start + numPolygonPoint };
        start += numPolygonPoint;
    }

    std::vector<VertId> flatPolygonIndices( faces.back().lastVertex );

    keepGoing = ParallelFor( faces, [&] ( size_t numPolygon )
    {
        if ( forseStop )
        {
            return;
        }
        size_t numLine = delta + numPolygon;

        const std::string_view line( &buf[splitLines[numLine]], splitLines[numLine + 1] - splitLines[numLine] );
        auto result = parsePolygon( line, &flatPolygonIndices[faces.vec_[numPolygon].firstVertex], nullptr );

        if ( !result.has_value() )
        {
            forseStop = true;
        }
    }, settings.callback );

    if ( forseStop )
    {
        return unexpected( std::string( "Error when reading polygon topology" ) );
    }
    if ( !keepGoing )
    {
        return unexpectedOperationCanceled();
    }

    FaceBitSet skippedFaces;
    MeshBuilder::BuildSettings buildSettings;
    if ( settings.skippedFaceCount )
    {
        skippedFaces.resize( faces.size(), true );
        buildSettings.region = &skippedFaces;
    }
    auto res = Mesh::fromFaceSoup( std::move( pointsBlocks ), flatPolygonIndices, faces, buildSettings );
    if ( settings.skippedFaceCount )
        *settings.skippedFaceCount = int( skippedFaces.count() );
    return res;
}

Expected<Mesh> fromObj( const std::filesystem::path & file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromObj( in, settings ), file );
}

Expected<Mesh> fromObj( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    auto objs = fromSceneObjFile( in, true, {}, settings );
    if ( !objs.has_value() )
        return unexpected( objs.error() );
    if ( objs->size() != 1 )
        return unexpected( "OBJ-file is empty" );

    return std::move( (*objs)[0].mesh );
}

Expected<MR::Mesh> fromAnyStl( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromAnyStl( in, settings ), file );
}

Expected<MR::Mesh> fromAnyStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    auto pos = in.tellg();
    auto resBin = fromBinaryStl( in, settings );
    if ( resBin.has_value() || resBin.error() == "Loading canceled" )
        return resBin;
    in.clear();
    in.seekg( pos );
    auto resAsc = fromASCIIStl( in, settings );
    if ( resAsc.has_value() )
        return resAsc;
    return unexpected( resBin.error() + '\n' + resAsc.error() );
}

Expected<Mesh> fromBinaryStl( const std::filesystem::path & file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromBinaryStl( in, settings ), file );
}

Expected<Mesh> fromBinaryStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    char header[80];
    in.read( header, 80 );

    std::uint32_t numTris;
    in.read( (char*)&numTris, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the number of triangles from STL-file" ) );

    auto posCur = in.tellg();
    in.seekg( 0, std::ios_base::end );
    auto posEnd = in.tellg();
    in.seekg( posCur );
    if ( posEnd - posCur < 50 * std::istream::pos_type( numTris ) )
        return unexpected( std::string( "Binary STL-file is too short" ) );

    MeshBuilder::VertexIdentifier vi;
    vi.reserve( numTris );

    #pragma pack(push, 1)
    struct StlTriangle
    {
        Vector3f normal;
        Vector3f vert[3];
        std::uint16_t attr;
    };
    #pragma pack(pop)
    static_assert( sizeof( StlTriangle ) == 50, "check your padding" );

    const auto itemsInBuffer = std::min( numTris, 32768u );
    std::vector<StlTriangle> buffer( itemsInBuffer ), nextBuffer( itemsInBuffer );
    std::vector<Triangle3f> chunk( itemsInBuffer );

    // first chunk
    in.read( (char*)buffer.data(), sizeof(StlTriangle) * itemsInBuffer );
    if ( !in  )
        return unexpected( std::string( "Binary STL read error" ) );

    size_t decodedBytes = 0;
    // 0.5 because fromTrianglesDuplicatingNonManifoldVertices takes at least half of time
    const float rStreamSize = 0.5f * sizeof( StlTriangle ) / float( posEnd - posCur );

    while ( !buffer.empty() )
    {
        // decode previously read buffer in a worked thread
        tbb::task_group taskGroup;
        taskGroup.run( [&chunk, &vi, &buffer] ()
        {
            chunk.resize( buffer.size() );
            for ( int i = 0; i < buffer.size(); ++i )
                for ( int j = 0; j < 3; ++j )
                    chunk[i][j] = buffer[i].vert[j];
            vi.addTriangles( chunk );
        } );

        if ( vi.numTris() + buffer.size() < numTris )
        {
            const auto itemsInNextChuck = std::min( numTris - (std::uint32_t)( vi.numTris() + buffer.size() ), itemsInBuffer );
            nextBuffer.resize( itemsInNextChuck );
            const size_t size = sizeof( StlTriangle ) * nextBuffer.size();
            // read from stream in the current thread to be compatible with PythonIstreamBuf
            in.read( ( char* )nextBuffer.data(), size );
        }
        else
            nextBuffer.clear();

        taskGroup.wait();
        decodedBytes += buffer.size();

        if ( !reportProgress( settings.callback , decodedBytes * rStreamSize ) )
            return unexpected( std::string( "Loading canceled" ) );
        if ( !in )
            return unexpected( std::string( "Binary STL read error" ) );
        buffer.swap( nextBuffer );
    }

//     #pragma warning(disable: 4244)
//     std::cout <<
//         "tris = " << numTris << "\n"
//         "verts = " << hmap.size() << "\n"
//         "bucket_count = " << hmap.bucket_count() << "\n"
//         "subcnt = " << hmap.subcnt() << "\n"
//         "load_factor = " << hmap.load_factor() << "\n"
//         "max_load_factor = " << hmap.max_load_factor() << "\n";

    auto t = vi.takeTriangulation();
    FaceBitSet skippedFaces;
    std::vector<MeshBuilder::VertDuplication> dups;
    std::vector<MeshBuilder::VertDuplication>* dupsPtr = nullptr;
    if ( settings.duplicatedVertexCount )
        dupsPtr = &dups;
    MeshBuilder::BuildSettings buildSettings;
    if ( settings.skippedFaceCount )
    {
        skippedFaces = FaceBitSet( t.size() );
        skippedFaces.set();
        buildSettings.region = &skippedFaces;
    }
    const auto res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( vi.takePoints(), t, dupsPtr, buildSettings );
    if ( settings.duplicatedVertexCount )
        *settings.duplicatedVertexCount = int( dups.size() );
    if ( settings.skippedFaceCount )
        *settings.skippedFaceCount = int( skippedFaces.count() );
    if ( !reportProgress( settings.callback , 1.0f ) )
        return unexpected( std::string( "Loading canceled" ) );
    return res;
}

Expected<Mesh> fromASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromASCIIStl( in, settings ), file );
}

Expected<Mesh> fromASCIIStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER;

    using HMap = ParallelHashMap<Vector3f, VertId>;
    HMap hmap;
    VertCoords points;
    Triangulation t;

    std::string line;
    std::string prefix;
    Vector3f point;
    ThreeVertIds currTri;
    int triPos = 0;
    bool solidFound = false;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    for ( int i = 0; std::getline( in, line ); ++i )
    {
        std::istringstream iss( line );
        if ( !( iss >> prefix ) )
            break;

        if ( !solidFound )
        {
            if ( prefix == "solid" )
                solidFound = true;
            else
                break;
        }

        if ( prefix == "outer" )
        {
            triPos = 0;
            continue;
        }
        if ( prefix == "vertex" )
        {
            double x, y, z; // double is used to correctly open coordinates like 1e-55 which are under of float-precision
            if ( !( iss >> x >> y >> z ) )
                break;
            point = Vector3f{ Vector3d{ x, y, z } };

            VertId& id = hmap[point];
            if ( !id.valid() )
            {
                id = VertId( points.size() );
                points.push_back( point );
            }
            currTri[triPos] = id;
            ++triPos;
            continue;
        }
        if ( prefix == "endloop" )
        {
            t.push_back( currTri );
            continue;
        }
        if ( settings.callback && !( i & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !settings.callback( progress ) )
                return unexpected( std::string( "Loading canceled" ) );
        }
    }

    if ( !solidFound )
        return unexpected( std::string( "Failed to find 'solid' prefix in ascii STL" ) );




    FaceBitSet skippedFaces;
    std::vector<MeshBuilder::VertDuplication> dups;
    std::vector<MeshBuilder::VertDuplication>* dupsPtr = nullptr;
    if ( settings.duplicatedVertexCount )
        dupsPtr = &dups;
    MeshBuilder::BuildSettings buildSettings;
    if ( settings.skippedFaceCount )
    {
        skippedFaces = FaceBitSet( t.size() );
        skippedFaces.set();
        buildSettings.region = &skippedFaces;
    }
    const auto res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( points ), t, dupsPtr, buildSettings );
    if ( settings.duplicatedVertexCount )
        *settings.duplicatedVertexCount = int( dups.size() );
    if ( settings.skippedFaceCount )
        *settings.skippedFaceCount = int( skippedFaces.count() );
    return res;
}

Expected<Mesh> fromPly( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPly( in, settings ), file );
}

Expected<Mesh> fromPly( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    const auto posStart = in.tellg();
    miniply::PLYReader reader( in );
    if ( !reader.valid() )
        return unexpected( std::string( "PLY file open error" ) );

    uint32_t indecies[3];
    bool gotVerts = false, gotFaces = false;

    std::vector<unsigned char> colorsBuffer;
    Mesh res;
    const auto posEnd = reader.get_end_pos();
    const float streamSize = float( posEnd - posStart );

    FaceBitSet skippedFaces;
    for ( int i = 0; reader.has_element() && ( !gotVerts || !gotFaces ); reader.next_element(), ++i )
    {
        if ( reader.element_is(miniply::kPLYVertexElement) && reader.load_element() )
        {
            auto numVerts = reader.num_rows();
            if ( reader.find_pos( indecies ) )
            {
                Timer t( "extractPoints" );
                res.points.resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, res.points.data() );
                gotVerts = true;
            }
            if ( settings.normals && reader.find_normal( indecies ) )
            {
                Timer t( "extractNormals" );
                settings.normals->resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, settings.normals->data() );
            }
            if ( settings.colors && reader.find_color( indecies ) )
            {
                Timer t( "extractColors" );
                colorsBuffer.resize( 3 * numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
            }
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !reportProgress( settings.callback, progress ) )
                return unexpected( std::string( "Loading canceled" ) );
            continue;
        }

        const auto posLast = in.tellg();
        if ( reader.element_is(miniply::kPLYFaceElement) && reader.load_element() && reader.find_indices(indecies) )
        {
            bool polys = reader.requires_triangulation( indecies[0] );
            if ( polys && !gotVerts )
                return unexpected( std::string( "PLY file open: need vertex positions to triangulate faces" ) );

            Triangulation tris;
            if (polys) 
            {
                Timer t( "extractTriangles" );
                auto numIndices = reader.num_triangles( indecies[0] );
                tris.resize( numIndices );
                reader.extract_triangles( indecies[0], &res.points.front().x, (std::uint32_t)res.points.size(), miniply::PLYPropertyType::Int, &tris.front() );
            }
            else 
            {
                Timer t( "extractTriples" );
                auto numIndices = reader.num_rows();
                tris.resize( numIndices );
                reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Int, &tris.front() );
            }
            const auto posCurent = in.tellg();
            // suppose  that reading is 10% of progress and building mesh is 90% of progress
            if ( !reportProgress( settings.callback, ( float( posLast ) + ( posCurent - posLast ) * 0.1f - posStart ) / streamSize ) )
                return unexpected( std::string( "Loading canceled" ) );
            bool isCanceled = false;
            ProgressCallback partedProgressCb = settings.callback ? [callback = settings.callback, posLast, posCurent, posStart, streamSize, &isCanceled] ( float v )
            {
                const bool res = callback( ( float( posLast ) + ( posCurent - posLast ) * ( 0.1f + v * 0.9f ) - posStart ) / streamSize );
                isCanceled |= !res;
                return res;
            } : settings.callback;

            MeshBuilder::BuildSettings buildSettings;
            if ( settings.skippedFaceCount )
            {
                skippedFaces = FaceBitSet( tris.size() );
                skippedFaces.set();
                buildSettings.region = &skippedFaces;
            }
            res.topology = MeshBuilder::fromTriangles( tris, buildSettings, partedProgressCb );
            if ( settings.skippedFaceCount )
                *settings.skippedFaceCount += int( skippedFaces.count() );
            if ( settings.callback && ( !settings.callback( float( posCurent - posStart ) / streamSize ) || isCanceled ) )
                return unexpected( std::string( "Loading canceled" ) );
            gotFaces = true;
        }
    }

    if ( !reader.valid() )
        return unexpected( std::string( "PLY file read or parse error" ) );

    if ( !gotVerts )
        return unexpected( std::string( "PLY file does not contain vertices" ) );

    if ( settings.colors && !colorsBuffer.empty() )
    {
        settings.colors->resize( res.points.size() );
        for ( VertId i{ 0 }; i < res.points.size(); ++i )
        {
            int ind = 3 * i;
            ( *settings.colors )[i] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
        }
    }

    return res;
}

#ifndef MRMESH_NO_OPENCTM

Expected<Mesh> fromCtm( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromCtm( in, settings ), file );
}

Expected<Mesh> fromCtm( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    class ScopedCtmConext 
    {
        CTMcontext context_ = ctmNewContext( CTM_IMPORT );
    public:
        ~ScopedCtmConext() { ctmFreeContext( context_ ); }
        operator CTMcontext() { return context_; }
    } context;


    struct LoadData
    {
        std::function<bool( float )> callbackFn{};
        std::istream* stream;
        bool wasCanceled{ false };
    } loadData;
    loadData.stream = &in;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );

    if ( settings.callback )
    {
        loadData.callbackFn = [callback = settings.callback, posStart, sizeAll = float( posEnd - posStart ), &in] ( float )
        {
            float progress = float( in.tellg() - posStart ) / sizeAll;
            return callback( progress );
        };
    }

    ctmLoadCustom( context, []( void * buf, CTMuint size, void * data )
    {
        LoadData& loadData = *reinterpret_cast<LoadData*>( data );
        auto& stream = *loadData.stream;
        auto pos = stream.tellg();
        loadData.wasCanceled |= !readByBlocks( stream, ( char* )buf, size, loadData.callbackFn, 1u << 12 );
        if ( loadData.wasCanceled )
            return 0u;
        return (CTMuint)( stream.tellg() - pos );
    }, &loadData );

    auto vertCount = ctmGetInteger( context, CTM_VERTEX_COUNT );
    auto triCount  = ctmGetInteger( context, CTM_TRIANGLE_COUNT );
    auto vertices  = ctmGetFloatArray( context, CTM_VERTICES );
    auto indices   = ctmGetIntegerArray( context, CTM_INDICES );
    if ( loadData.wasCanceled )
        return unexpected( "Loading canceled" );
    if ( ctmGetError(context) != CTM_NONE )
        return unexpected( "Error reading CTM format" );

    // even if we save false triangle (0,0,0) in MG2 format, it can be open as triangle (i,i,i)
    if ( triCount == 1 && indices[0] == indices[1] && indices[0] == indices[2] )
    {
        // CTM file is representing points, but it was written with the library requiring the presence of at least one triangle
        triCount = 0;
    }

    if ( settings.colors )
    {
        auto colorAttrib = ctmGetNamedAttribMap( context, "Color" );
        if ( colorAttrib != CTM_NONE )
        {
            auto colorArray = ctmGetFloatArray( context, colorAttrib );
            settings.colors->resize( vertCount );
            for ( VertId i{ 0 }; CTMuint( i ) < vertCount; ++i )
            {
                auto j = 4 * i;
                (*settings.colors)[i] = Color( colorArray[j], colorArray[j + 1], colorArray[j + 2], colorArray[j + 3] );
            }
        }
    }

    if ( settings.normals && ctmGetInteger( context, CTM_HAS_NORMALS ) == CTM_TRUE )
    {
        auto normals = ctmGetFloatArray( context, CTM_NORMALS );
        settings.normals->resize( vertCount );
        for ( VertId i{0}; i < (int) vertCount; ++i )
            (*settings.normals)[i] = Vector3f( normals[3 * i], normals[3 * i + 1], normals[3 * i + 2] );
    }

    Mesh mesh;
    mesh.points.resize( vertCount );
    for ( VertId i{0}; i < (int)vertCount; ++i )
        mesh.points[i] = Vector3f( vertices[3*i], vertices[3*i+1], vertices[3*i+2] );

    Triangulation t;
    t.reserve( triCount );
    for ( FaceId i{0}; i < (int)triCount; ++i )
        t.push_back( { VertId( (int)indices[3*i] ), VertId( (int)indices[3*i+1] ), VertId( (int)indices[3*i+2] ) } );

    FaceBitSet skippedFaces;
    MeshBuilder::BuildSettings buildSettings;
    if ( settings.skippedFaceCount )
    {
        skippedFaces = FaceBitSet( t.size() );
        skippedFaces.set();
        buildSettings.region = &skippedFaces;
    }
    mesh.topology = MeshBuilder::fromTriangles( t, buildSettings );
    if ( settings.skippedFaceCount )
        *settings.skippedFaceCount = int( skippedFaces.count() );

    return mesh;
}
#endif

Expected<Mesh> fromDxf( const std::filesystem::path& path, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( path, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( path ) );

    return addFileNameInError( fromDxf( in, settings ), path );
}

Expected<Mesh> fromDxf( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    // find size
    in.seekg( 0, std::ios_base::end );
    const size_t size = in.tellg();
    in.seekg( 0 );

    std::vector<Triangle3f> triangles;
    std::string str;
    std::getline( in, str );

    int code = {};
    if ( !parseSingleNumber<int>( str, code ) )
        return unexpected( "File is corrupted" );
    
    bool is3DfaceFound = false;

    for ( int i = 0; !in.eof(); ++i )
    {
        if ( i % 1024 == 0 && !reportProgress( settings.callback, float( in.tellg() ) / size ) )
            return unexpectedOperationCanceled();
        
        std::getline( in, str );

        if ( str == "3DFACE" )
        {
            triangles.emplace_back();
            is3DfaceFound = true;
        }

        if ( is3DfaceFound )
        {
            const int vIdx = code % 10;
            const int cIdx = code / 10 - 1;
            if ( vIdx >= 0 && vIdx < 3 && cIdx >= 0 && cIdx < 3 )
            {
                if ( !parseSingleNumber<float>( str, triangles.back()[vIdx][cIdx] ) )
                    return unexpected( "File is corrupted" );
            }
        }

        std::getline( in, str );
        if ( str.empty() )
            continue;
        
        if ( !parseSingleNumber<int>( str, code ) )
            return unexpected( "File is corrupted" );

        if ( code == 0 )
            is3DfaceFound = false;
    }

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( triangles.empty() )
        return unexpected( "No mesh is found" );

    return Mesh::fromPointTriples( triangles, true );
}

Expected<Mesh> fromAnySupportedFormat( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char)tolower( c );

    ext = "*" + ext;

    Expected<MR::Mesh> res = unexpected( std::string( "unsupported file extension" ) );
    auto filters = getFilters();
    auto itF = std::find_if( filters.begin(), filters.end(), [ext]( const IOFilter& filter )
    {
        return filter.extensions.find( ext ) != std::string::npos;
    } );
    if ( itF == filters.end() )
        return res;

    auto loader = getMeshLoader( *itF );
    if ( !loader )
        return res;
    return loader( file, settings );
}

Expected<Mesh> fromAnySupportedFormat( std::istream& in, const std::string& extension, const MeshLoadSettings& settings /*= {}*/ )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<MR::Mesh> res = unexpected( std::string( "unsupported file extension" ) );
    auto filters = getFilters();
    auto itF = std::find_if( filters.begin(), filters.end(), [ext] ( const IOFilter& filter )
    {
        return filter.extensions.find( ext ) != std::string::npos;
    } );
    if ( itF == filters.end() )
        return res;

    auto loader = getMeshStreamLoader( *itF );
    if ( !loader )
        return res;

    return loader( in, settings );
}

/*
MeshLoaderAdder __meshLoaderAdder( NamedMeshLoader{IOFilter( "MrMesh (.mrmesh)", "*.mrmesh" ),MeshLoader{static_cast<Expected<MR::Mesh>(*)(const std::filesystem::path&,VertColors*)>(fromMrmesh)}} );
*/

MR_ADD_MESH_LOADER( IOFilter( "MeshInspector (.mrmesh)", "*.mrmesh" ), fromMrmesh )
MR_ADD_MESH_LOADER( IOFilter( "Stereolithography (.stl)", "*.stl" ), fromAnyStl )
MR_ADD_MESH_LOADER( IOFilter( "Object format file (.off)", "*.off" ), fromOff )
MR_ADD_MESH_LOADER( IOFilter( "3D model object (.obj)", "*.obj" ), fromObj )
MR_ADD_MESH_LOADER( IOFilter( "Polygon File Format (.ply)", "*.ply" ), fromPly )
MR_ADD_MESH_LOADER( IOFilter( "Drawing Interchange Format (.dxf)", "*.dxf" ), fromDxf )
#ifndef MRMESH_NO_OPENCTM
MR_ADD_MESH_LOADER( IOFilter( "Compact triangle-based mesh (.ctm)", "*.ctm" ), fromCtm )
#endif
#ifndef MRMESH_NO_OPENCASCADE
MR_ADD_MESH_LOADER( IOFilter( "STEP files (.step,.stp)", "*.step;*.stp" ), fromStep )
#endif

} //namespace MeshLoad

} //namespace MR
