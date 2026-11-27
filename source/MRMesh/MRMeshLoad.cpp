#include "MRMeshLoad.h"
#include "MRBitCast.h"
#include "MRMeshBuilder.h"
#include "MRIdentifyVertices.h"
#include "MRMesh.h"
#include "MRphmap.h"
#include "MRTimer.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"
#include "MRMeshLoadObj.h"
#include "MRObjectMesh.h"
#include "MRObjectsAccess.h"
#include "MRColor.h"
#include "MRProgressReadWrite.h"
#include "MRIOParsing.h"
#include "MRMeshDelone.h"
#include "MRPly.h"
#include "MRTriMesh.h"
#include "MRFaceSoup.h"
#include "MRParallelFor.h"
#include "MRImageLoad.h"
#include "MRTelemetry.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRTBB.h"
#include "MRBitSetParallelFor.h"
#include "MRString.h"

#include <array>
#include <future>
#include <sstream>
#include <fstream>

namespace MR
{

void telemetryStlHead( const char* prefix, std::string s );

Expected<Mesh> loadMrmesh( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromMrmesh( file, settings );
}

Expected<Mesh> loadMrmesh( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromMrmesh( in, settings );
}

Expected<Mesh> loadOff( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromOff( file, settings );
}

Expected<Mesh> loadOff( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromOff( in, settings );
}

Expected<Mesh> loadObj( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromObj( file, settings );
}

Expected<Mesh> loadObj( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromObj( in, settings );
}

Expected<Mesh> loadStl( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromAnyStl( file, settings );
}

Expected<Mesh> loadStl( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromAnyStl( in, settings );
}

Expected<Mesh> loadBinaryStl( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromBinaryStl( file, settings );
}

Expected<Mesh> loadBinaryStl( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromBinaryStl( in, settings );
}

Expected<Mesh> loadASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromASCIIStl( file, settings );
}

Expected<Mesh> loadASCIIStl( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromASCIIStl( in, settings );
}

/// the fraction of progress for loading TriMesh or FaceSoup in most formats, the rest is for building Mesh from it
constexpr float cLoadAsTriMeshProgress = 0.7f;

/// settings for loading TriMesh or FaceSoup, which takes given fraction of progress before building Mesh from it
static MeshLoadSettings triMeshSettings( const MeshLoadSettings& settings, float triMeshProgress )
{
    auto res = settings;
    res.callback = subprogress( settings.callback, 0.0f, triMeshProgress );
    return res;
}

/// builds Mesh from just loaded TriMesh, resolving non-manifold vertices by their duplication together with their attributes
static Expected<Mesh> triMeshToMesh( Expected<TriMesh> && triMesh, const MeshLoadSettings& settings )
{
    if ( !triMesh )
        return unexpected( std::move( triMesh.error() ) );
    const auto numLoadedPoints = triMesh->points.size();
    std::vector<MeshBuilder::VertDuplication> dups;
    auto res = Mesh::fromTriMeshDuplicatingNonManifoldVertices( std::move( *triMesh ), &dups, { .skippedFaceCount = settings.skippedFaceCount } );
    if ( settings.duplicatedVertexCount )
        *settings.duplicatedVertexCount = int( dups.size() );
    // the attributes of other sizes were not filled by the loader
    auto copyDupAttributes = [&] ( auto * attributes )
    {
        if ( attributes )
            MeshBuilder::copyAttributesToDuplicates( *attributes, dups, numLoadedPoints, res.points.size() );
    };
    copyDupAttributes( settings.colors );
    copyDupAttributes( settings.uvCoords );
    copyDupAttributes( settings.normals );
    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

/// checks that all vertex ids of the polygons are in the range of points
static Expected<FaceSoup> checkFaceSoup( FaceSoup && soup )
{
    for ( auto v : soup.verts )
        if ( !v || v >= (int)soup.points.size() )
            return unexpected( "vertex id is out of range of point coordinates" );
    return std::move( soup );
}

/// builds Mesh from just loaded FaceSoup, filling its polygons
static Expected<Mesh> faceSoupToMesh( Expected<FaceSoup> && soup, const MeshLoadSettings& settings )
{
    if ( !soup )
        return unexpected( std::move( soup.error() ) );
    return Mesh::fromFaceSoup( std::move( *soup ),
        { .skippedFaceCount = settings.skippedFaceCount }, subprogress( settings.callback, cLoadAsTriMeshProgress, 1.0f ) );
}

// object color by Materialise Magics convention: "COLOR=" followed by red, green, blue, alpha bytes anywhere in the header
static std::optional<Color> stlHeaderColor( std::string_view header )
{
    const std::string_view key = "COLOR=";
    const auto n = header.find( key );
    if ( n == std::string_view::npos || n + key.size() + 4 > header.size() )
        return {};
    const auto c = [&]( size_t i ) { return int( (unsigned char)header[n + key.size() + i] ); };
    return Color( c( 0 ), c( 1 ), c( 2 ), c( 3 ) );
}

Expected<TriMesh> loadBinaryStlAsTriMesh( std::istream& in, const MeshLoadSettings& settings )
{
    MR_TIMER;

    char header[81];
    in.read( header, 80 );
    header[80] = '\0';
    if ( settings.solidColor )
        if ( auto c = stlHeaderColor( std::string_view( header, 80 ) ) )
            *settings.solidColor = c;

    std::uint32_t numTris;
    in.read( (char*)&numTris, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the number of triangles from STL-file" ) );

    const auto streamSize = getStreamSize( in );
    if ( streamSize < 50 * std::istream::pos_type( numTris ) )
        return unexpected( std::string( "Binary STL-file is too short" ) );

    MeshBuilder::VertexIdentifier vi;
    vi.reserve( numTris );

    using Pos3f = std::array<char, 12>;
    struct StlTriangle
    {
        Pos3f normal;
        Pos3f coords[3];
        char attrs[2];
        // floats in Vector3f must be 4-bytes aligned on some platforms, so we use chars and cast them in Vector3f on access
        Vector3f vertex( int i ) const
        {
            return MR::bit_cast<Vector3f>( coords[i] );
        }
    };
    static_assert( sizeof( StlTriangle ) == 50 );
    static_assert( alignof( StlTriangle ) <= 2 );

    const auto itemsInBuffer = std::min( numTris, 32768u );
    std::vector<StlTriangle> buffer( itemsInBuffer ), nextBuffer( itemsInBuffer );
    std::vector<Triangle3f> chunk( itemsInBuffer );

    // first chunk
    in.read( (char*)buffer.data(), sizeof(StlTriangle) * itemsInBuffer );
    if ( !in  )
        return unexpected( std::string( "Binary STL read error" ) );

    size_t decodedBytes = 0;
    const float rStreamSize = sizeof( StlTriangle ) / float( streamSize );

    while ( !buffer.empty() )
    {
        auto preNumTris = vi.numTris(); // must be read before the task below starts adding more triangles in (vi)

        // decode previously read buffer in a worked thread
        tbb::task_group taskGroup;
        taskGroup.run( [&chunk, &vi, &buffer] ()
        {
            chunk.resize( buffer.size() );
            for ( int i = 0; i < buffer.size(); ++i )
                for ( int j = 0; j < 3; ++j )
                    chunk[i][j] = buffer[i].vertex( j );
            vi.addTriangles( chunk );
        } );

        if ( preNumTris + buffer.size() < numTris )
        {
            const auto itemsInNextChuck = std::min( numTris - (std::uint32_t)( preNumTris + buffer.size() ), itemsInBuffer );
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
            return unexpectedOperationCanceled();
        if ( !in )
            return unexpected( std::string( "Binary STL read error" ) );
        buffer.swap( nextBuffer );
    }

    TriMesh res;
    res.tris = vi.takeTriangulation();
    res.points = vi.takePoints();
    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( settings.telemetrySignal )
        telemetryStlHead( "STL head ", header );

    return res;
}

Expected<TriMesh> loadASCIIStlAsTriMesh( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( loadASCIIStlAsTriMesh( in, settings ), file );
}

Expected<TriMesh> loadASCIIStlAsTriMesh( std::istream& in, const MeshLoadSettings& settings )
{
    MR_TIMER;

    auto data = readCharBuffer( in );
    if ( !data.has_value() )
        return unexpected( data.error() );

    std::string_view dataView( data->data(), data->size() );

    if ( hasBom( dataView ) )
        dataView = dataView.substr( 3 );

    const auto newlines = splitByLines( dataView.data(), dataView.size() );

    if ( !reportProgress( settings.callback, 0.15f ) )
        return unexpectedOperationCanceled();

    if ( newlines.size() < 2 )
        return unexpected( std::string( "Ascii STL is too short" ) );

    auto headerView = dataView.substr( newlines[0], newlines[1] );
    if ( !headerView.starts_with( "solid" ) )
        return unexpected( std::string( "Failed to find 'solid' prefix in ascii STL" ) );

    BitSet faceRepresentativeLines( newlines.size() - 1 );
    auto keepGoing = BitSetParallelForAll( faceRepresentativeLines, [&] ( size_t lineI )
    {
        auto startLine = newlines[lineI];
        auto size = newlines[lineI + 1] - newlines[lineI];
        if ( trimLeft( dataView.substr( startLine, size ) ).starts_with( "outer" ) )
            faceRepresentativeLines.set( lineI );
    }, subprogress( settings.callback, 0.15f, 0.3f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    auto numTris = faceRepresentativeLines.count();
    const auto itemsInBuffer = std::min( numTris, size_t( 32768 ) );
    const int numChunks = itemsInBuffer > 0 ? ( int( numTris ) + int( itemsInBuffer ) - 1 ) / int( itemsInBuffer ) : 0;
    std::vector<Triangle3f> chunk( itemsInBuffer );
    std::vector<Triangle3f> chunk2( itemsInBuffer );

    auto lastFaceLine = faceRepresentativeLines.find_last();

    MeshBuilder::VertexIdentifier vi;
    vi.reserve( numTris );
    tbb::task_group taskGroup;

    int i = 0;
    int processedChunks = 0;
    auto sb = subprogress( settings.callback, 0.3f, 0.9f );
    for ( auto fLine : faceRepresentativeLines )
    {
        if ( i < itemsInBuffer )
        {
            int triI = 0;
            Triangle3f tri;
            for ( auto lineI = fLine + 1; lineI + 1 < newlines.size(); ++lineI )
            {
                if ( faceRepresentativeLines.test( lineI ) )
                    break;
                auto lineView = trimLeft( dataView.substr( newlines[lineI], newlines[lineI + 1] - newlines[lineI] ) );
                if ( lineView.starts_with( "endloop" ) )
                    break;
                if ( !lineView.starts_with( "vertex" ) )
                    continue;
                if ( triI == 3 )
                    return unexpected( "Polygonal STL is not supported, line: " + std::to_string( newlines[lineI] ) );
                auto parseRes = parseTextCoordinate( lineView.substr( 7 ), tri[triI++] );
                if ( !parseRes.has_value() )
                    return unexpected( parseRes.error() + " line: " + std::to_string( newlines[lineI] ) );
            }
            if ( triI != 3 )
                return unexpected( "Too few vertices for triangle, line: " + std::to_string( newlines[fLine] ) );
            chunk[i++] = std::move( tri );
        }
        if ( i == itemsInBuffer || fLine == lastFaceLine )
        {
            taskGroup.wait();
            if ( !reportProgress( sb, float( ++processedChunks ) / float( numChunks ) ) )
                return unexpectedOperationCanceled();
            std::swap( chunk, chunk2 );
            chunk2.resize( i );
            i = 0;
            if ( fLine != lastFaceLine ) // add previous chunk in indexer in parallel
                taskGroup.run( [&chunk2, &vi] () { vi.addTriangles( chunk2 ); } );
            else
                vi.addTriangles( chunk2 );
        }
    }

    TriMesh res;
    res.tris = vi.takeTriangulation();
    res.points = vi.takePoints();
    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( settings.telemetrySignal )
        telemetryStlHead( "STL ASCII ", std::string( headerView.substr( 6 ) ) ); // "solid " - 6 chars

    return res;
}

Expected<Mesh> loadPly( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromPly( file, settings );
}

Expected<Mesh> loadPly( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromPly( in, settings );
}

Expected<Mesh> loadDxf( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromDxf( file, settings );
}

Expected<Mesh> loadDxf( std::istream& in, const MeshLoadSettings& settings )
{
    return MeshLoad::fromDxf( in, settings );
}

Expected<Mesh> loadMesh( const std::filesystem::path& file, const MeshLoadSettings& settings )
{
    return MeshLoad::fromAnySupportedFormat( file, settings );
}

Expected<Mesh> loadMesh( std::istream& in, const std::string& extension, const MeshLoadSettings& settings )
{
    return MeshLoad::fromAnySupportedFormat( in, extension, settings );
}

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
    MR_TIMER;

    Mesh mesh;
    auto readRes = mesh.topology.read( in, subprogress( settings.callback, 0.f, 0.5f) );
    if ( !readRes.has_value() )
    {
        std::string error = readRes.error();
        if ( error != stringOperationCanceled() )
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
        return unexpectedOperationCanceled();

    if ( !in )
        return unexpected( std::string( "Error reading  points from mrmesh-file" ) );

    return mesh;
}

Expected<Mesh> fromOff( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    return faceSoupToMesh( fromOffAsFaceSoup( file, triMeshSettings( settings, cLoadAsTriMeshProgress ) ), settings );
}

Expected<Mesh> fromOff( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return faceSoupToMesh( fromOffAsFaceSoup( in, triMeshSettings( settings, cLoadAsTriMeshProgress ) ), settings );
}

Expected<FaceSoup> fromOffAsFaceSoup( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromOffAsFaceSoup( in, settings ), file );
}

/// binary OFF: big-endian 32-bit integers and floats following the newline after BINARY keyword
static Expected<FaceSoup> fromOffBinary( std::istream& in, const MeshLoadSettings& settings )
{
    std::string restOfHeaderLine;
    std::getline( in, restOfHeaderLine );

    auto bufOrExpect = readCharBuffer( in );
    if ( !bufOrExpect )
        return unexpected( std::move( bufOrExpect.error() ) );
    const auto* data = reinterpret_cast<const unsigned char*>( bufOrExpect->data() );
    const size_t size = bufOrExpect->size();

    auto get32 = [data] ( size_t pos )
    {
        const auto* p = data + pos;
        return ( std::uint32_t( p[0] ) << 24 ) | ( std::uint32_t( p[1] ) << 16 ) | ( std::uint32_t( p[2] ) << 8 ) | std::uint32_t( p[3] );
    };
    size_t pos = 0;
    auto readInt = [&] ( int& v )
    {
        if ( size - pos < 4 )
            return false;
        v = bit_cast<int>( get32( pos ) );
        pos += 4;
        return true;
    };

    int numPoints = 0, numPolygons = 0, numEdges = 0;
    if ( !readInt( numPoints ) || !readInt( numPolygons ) || !readInt( numEdges ) )
        return unexpected( std::string( "Unexpected end of binary OFF-file" ) );
    if ( numPoints <= 0 || numPolygons <= 0 )
        return unexpected( std::string( "Unsupported OFF-format" ) );
    // each polygon takes at least two integers: the number of vertices and the number of color components
    if ( ( size - pos ) / 12 < size_t( numPoints ) || ( size - pos - 12 * size_t( numPoints ) ) / 8 < size_t( numPolygons ) )
        return unexpected( std::string( "Unexpected end of binary OFF-file" ) );

    VertCoords points( numPoints );
    if ( !ParallelFor( points, [&] ( size_t i )
    {
        const size_t p = pos + 12 * i;
        points[VertId( i )] = Vector3f( bit_cast<float>( get32( p ) ), bit_cast<float>( get32( p + 4 ) ), bit_cast<float>( get32( p + 8 ) ) );
    }, subprogress( settings.callback, 0.0f, 0.4f ) ) )
        return unexpectedOperationCanceled();
    pos += 12 * size_t( numPoints );

    Vector<MeshBuilder::VertSpan, FaceId> faces( numPolygons );
    Vector<VertId> flatPolygonIndices;
    flatPolygonIndices.reserve( 3 * size_t( numPolygons ) );
    auto sp = subprogress( settings.callback, 0.4f, 1.0f );
    for ( int f = 0; f < numPolygons; ++f )
    {
        int numPolygonPoints = 0;
        if ( !readInt( numPolygonPoints ) )
            return unexpected( std::string( "Unexpected end of binary OFF-file" ) );
        if ( numPolygonPoints < 0 || ( size - pos ) / 4 < size_t( numPolygonPoints ) + 1 )
            return unexpected( std::string( "Error when reading polygon topology" ) );
        const int firstVertex = int( flatPolygonIndices.size() );
        for ( int i = 0; i < numPolygonPoints; ++i )
        {
            int v = 0;
            readInt( v );
            flatPolygonIndices.emplace_back( v );
        }
        faces[FaceId( f )] = { firstVertex, firstVertex + numPolygonPoints };

        // per-face color components are skipped
        int numColorComponents = 0;
        readInt( numColorComponents );
        if ( numColorComponents < 0 || ( size - pos ) / 4 < size_t( numColorComponents ) )
            return unexpected( std::string( "Unexpected end of binary OFF-file" ) );
        pos += 4 * size_t( numColorComponents );

        if ( !reportProgress( sp, [&] { return float( f ) / numPolygons; }, f, 0x10000 ) )
            return unexpectedOperationCanceled();
    }

    return checkFaceSoup( { std::move( points ), std::move( flatPolygonIndices ), std::move( faces ) } );
}

Expected<FaceSoup> fromOffAsFaceSoup( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER;

    std::string header;
    in >> header;
    if ( hasBom( header ) )
        header = header.substr( 3 );
    if ( !in || header != "OFF" )
        return unexpected( std::string( "File is not in OFF-format" ) );
    // some options are not supported yet: http://www.geomview.org/docs/html/OFF.html

    while ( in.peek() == ' ' || in.peek() == '\t' )
        in.get();
    if ( in.peek() == 'B' )
    {
        std::string format;
        in >> format;
        if ( format != "BINARY" )
            return unexpected( std::string( "Unsupported OFF-format" ) );
        return fromOffBinary( in, settings );
    }

    int numPoints, numPolygons, numUnused;
    in >> numPoints >> numPolygons >> numUnused;
    if ( !in || numPoints <= 0 || numPolygons <= 0 || numUnused != 0 )
        return unexpected( std::string( "Unsupported OFF-format" ) );

    auto bufOrExpect = readCharBuffer( in );
    if ( !bufOrExpect )
        return unexpected( std::move( bufOrExpect.error() ) );
    auto& buf = bufOrExpect.value();

    auto splitLines = splitByLines( buf.data(), buf.size() );

    size_t strHeader = 0;
    for ( size_t i = 0; i < splitLines.size(); i++ )
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

    VertCoords pointsBlocks( numPoints );

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
        pointsBlocks[VertId( numPoint )] = Vector3f( temp );

        if ( !result.has_value() )
        {
            forseStop = true;
        }
    }, subprogress( settings.callback, 0.0f, 0.4f ) );

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
        if ( auto e = parseFirstNum( line, numPolygonPoint ); !e )
            return unexpected( std::move( e.error() ) );

        faces[FaceId( i )] = MeshBuilder::VertSpan{ start, start + numPolygonPoint };
        start += numPolygonPoint;
    }
    if ( !reportProgress( settings.callback, 0.5f ) )
        return unexpectedOperationCanceled();

    Vector<VertId> flatPolygonIndices( faces.back().lastVertex );

    keepGoing = ParallelFor( faces, [&] ( size_t numPolygon )
    {
        if ( forseStop )
        {
            return;
        }
        size_t numLine = delta + numPolygon;

        const std::string_view line( &buf[splitLines[numLine]], splitLines[numLine + 1] - splitLines[numLine] );
        auto result = parsePolygon( line, flatPolygonIndices.data() + faces[FaceId( numPolygon )].firstVertex, nullptr );

        if ( !result.has_value() )
        {
            forseStop = true;
        }
    }, subprogress( settings.callback, 0.5f, 1.0f ) );

    if ( forseStop )
    {
        return unexpected( std::string( "Error when reading polygon topology" ) );
    }
    if ( !keepGoing )
    {
        return unexpectedOperationCanceled();
    }

    return checkFaceSoup( { std::move( pointsBlocks ), std::move( flatPolygonIndices ), std::move( faces ) } );
}

Expected<Mesh> fromObj( const std::filesystem::path & file, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromObjAsTriMesh( file, triMeshSettings( settings, cLoadAsTriMeshProgress ) ), settings );
}

Expected<Mesh> fromObj( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromObjAsTriMesh( in, triMeshSettings( settings, cLoadAsTriMeshProgress ) ), settings );
}

Expected<Mesh> fromAnyStl( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromAnyStlAsTriMesh( file, triMeshSettings( settings, cLoadAsTriMeshProgress ) ), settings );
}

Expected<Mesh> fromAnyStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromAnyStlAsTriMesh( in, triMeshSettings( settings, cLoadAsTriMeshProgress ) ), settings );
}

Expected<TriMesh> fromAnyStlAsTriMesh( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromAnyStlAsTriMesh( in, settings ), file );
}

Expected<TriMesh> fromAnyStlAsTriMesh( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    auto pos = in.tellg();
    auto resBin = fromBinaryStlAsTriMesh( in, settings );
    if ( resBin.has_value() || resBin.error() == stringOperationCanceled() )
        return resBin;
    in.clear();
    in.seekg( pos );
    auto resAsc = fromASCIIStlAsTriMesh( in, settings );
    if ( resAsc.has_value() )
        return resAsc;
    return unexpected( resBin.error() + '\n' + resAsc.error() );
}

// 0.5 because fromTrianglesDuplicatingNonManifoldVertices takes at least half of time
Expected<Mesh> fromBinaryStl( const std::filesystem::path & file, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromBinaryStlAsTriMesh( file, triMeshSettings( settings, 0.5f ) ), settings );
}

Expected<Mesh> fromBinaryStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromBinaryStlAsTriMesh( in, triMeshSettings( settings, 0.5f ) ), settings );
}

Expected<TriMesh> fromBinaryStlAsTriMesh( const std::filesystem::path & file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromBinaryStlAsTriMesh( in, settings ), file );
}

Expected<TriMesh> fromBinaryStlAsTriMesh( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return loadBinaryStlAsTriMesh( in, settings );
}

Expected<Mesh> fromASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromASCIIStlAsTriMesh( file, triMeshSettings( settings, 0.9f ) ), settings );
}

Expected<Mesh> fromASCIIStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromASCIIStlAsTriMesh( in, triMeshSettings( settings, 0.9f ) ), settings );
}

Expected<TriMesh> fromASCIIStlAsTriMesh( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    return loadASCIIStlAsTriMesh( file, settings );
}

Expected<TriMesh> fromASCIIStlAsTriMesh( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return loadASCIIStlAsTriMesh( in, settings );
}

static Expected<TriMesh> fromPlyAsTriMesh( std::istream& in, const MeshLoadSettings& settings, const std::filesystem::path& dir )
{
    MR_TIMER;

    std::optional<Triangulation> tris;
    PlyLoadParams params =
    {
        .tris = &tris,
        .edges = settings.edges,
        .colors = settings.colors,
        .faceColors = settings.faceColors,
        .uvCoords = settings.uvCoords,
        .triCornerUvCoords = settings.triCornerUvCoords,
        .normals = settings.normals,
        .texture = settings.texture,
        .dir = dir,
        .callback = subprogress( settings.callback, 0.0f, 0.9f ),
        .telemetrySignal = settings.telemetrySignal
    };
    auto maybePoints = loadPly( in, params );
    if ( !maybePoints )
        return unexpected( std::move( maybePoints.error() ) );

    TriMesh res;
    res.points = std::move( *maybePoints );

    if ( tris )
    {
        for ( auto & t : *tris )
        {
            for ( auto v : t )
                if ( v >= (int)res.points.size() )
                    return unexpected( "vertex id is larger than total point coordinates" );
            if ( !t[0] || !t[1] || !t[2] )
                t = {}; // invalid triangle, skipped when building Mesh
        }
        res.tris = std::move( *tris );
    }

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

// suppose that reading is 10% of progress and building mesh is 90% of progress
Expected<Mesh> fromPly( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromPlyAsTriMesh( file, triMeshSettings( settings, 0.1f ) ), settings );
}

Expected<Mesh> fromPly( std::istream& in, const MeshLoadSettings& settings )
{
    return triMeshToMesh( fromPlyAsTriMesh( in, triMeshSettings( settings, 0.1f ) ), settings );
}

Expected<TriMesh> fromPlyAsTriMesh( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPlyAsTriMesh( in, settings, file.parent_path() ), file );
}

Expected<TriMesh> fromPlyAsTriMesh( std::istream& in, const MeshLoadSettings& settings )
{
    return fromPlyAsTriMesh( in, settings, std::filesystem::path{} );
}

Expected<Mesh> fromDxf( const std::filesystem::path& path, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromDxfAsTriMesh( path, triMeshSettings( settings, 0.9f ) ), settings );
}

Expected<Mesh> fromDxf( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    return triMeshToMesh( fromDxfAsTriMesh( in, triMeshSettings( settings, 0.9f ) ), settings );
}

Expected<TriMesh> fromDxfAsTriMesh( const std::filesystem::path& path, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( path, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( path ) );

    return addFileNameInError( fromDxfAsTriMesh( in, settings ), path );
}

Expected<TriMesh> fromDxfAsTriMesh( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    // find size
    const auto posStart = in.tellg();
    const auto size = getStreamSize( in );

    std::vector<Triangle3f> triangles;
    std::string str;
    std::getline( in, str );
    if ( hasBom( str ) )
        str = str.substr( 3 );

    int code = {};
    if ( !parseSingleNumber<int>( str, code ) )
        return unexpected( "File is corrupted" );

    bool is3DfaceFound = false;

    for ( int i = 0; !in.eof(); ++i )
    {
        if ( i % 1024 == 0 && !reportProgress( settings.callback, float( in.tellg() - posStart ) / float( size ) ) )
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

    if ( triangles.empty() )
        return unexpected( stringUnsupportedFileFormat() + ": no mesh is found in DXF file" ); // lines or points can be there

    MeshBuilder::VertexIdentifier vi;
    vi.reserve( triangles.size() );
    vi.addTriangles( triangles );
    TriMesh res;
    res.tris = vi.takeTriangulation();
    res.points = vi.takePoints();

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

void telemetryLogSize( const Mesh& mesh )
{
    if ( int logFaces = intLog2( mesh.topology.numValidFaces() ) )
        TelemetrySignal( "Open Mesh Log Tris " + std::to_string( logFaces ) );
    else if ( int logPoints = intLog2( (int)mesh.points.size() ) ) // if opened file contains no triangles but only points
        TelemetrySignal( "Open Mesh Log Pnts " + std::to_string( logPoints ) );
}

static void telemetryOpenMesh( const std::string& ext, const Mesh& mesh, const MeshLoadSettings& settings )
{
    if ( !settings.telemetrySignal )
        return;

    std::string signalString = "Open " + ext;

    if ( auto lv = mesh.points.size() ) // not lv = mesh.topology.lastValidVert(), since topology can be empty
    {
        signalString += " VP";
        if ( settings.normals && settings.normals->size() >= lv )
            signalString += 'N';
        if ( settings.colors && settings.colors->size() >= lv )
            signalString += 'C';
        if ( settings.uvCoords && settings.uvCoords->size() >= lv )
            signalString += "UV";
    }

    if ( auto lf = mesh.topology.lastValidFace() )
    {
        signalString += " TRI";
        if ( settings.faceColors && settings.faceColors->size() >= lf )
            signalString += 'C';
    }

    if ( settings.texture && !settings.texture->pixels.empty() )
        signalString += " TEX";

    if ( settings.skippedFaceCount && *settings.skippedFaceCount > 0 )
        signalString += " SKIPF";

    if ( settings.duplicatedVertexCount && *settings.duplicatedVertexCount > 0 )
        signalString += " DUPV";

    if ( settings.xf && *settings.xf != AffineXf3f{} )
        signalString += " XF";

    TelemetrySignal( signalString );
    telemetryLogSize( mesh );
}

Expected<Mesh> fromAnySupportedFormat( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char)tolower( c );
    ext = "*" + ext;

    auto loader = getMeshLoader( ext );
    if ( !loader.fileLoad )
    {
        if ( loader.streamLoad )
        {
            std::ifstream in( file, std::ifstream::binary );
            if ( !in )
                return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );
            return addFileNameInError( loader.streamLoad( in, settings ), file );
        }
        // the error string must start with stringUnsupportedFileExtension()
        std::string err = fmt::format( "{} {} for mesh loading.", stringUnsupportedFileExtension(), ext );
        if ( SceneLoad::getSceneLoader( ext ) )
            return unexpected( err + "\nPlease open this format using scene loading function." );
        return unexpected( err );
    }

    auto res = loader.fileLoad( file, settings );
    if ( res )
        telemetryOpenMesh( ext, *res, settings );
    return res;
}

Expected<Mesh> fromAnySupportedFormat( std::istream& in, const std::string& extension, const MeshLoadSettings& settings /*= {}*/ )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto loader = getMeshLoader( ext );
    if ( !loader.streamLoad )
    {
        // the error string must start with stringUnsupportedFileExtension()
        std::string err = fmt::format( "{} {} for mesh loading.", stringUnsupportedFileExtension(), ext );
        if ( loader.fileLoad )
            return unexpected( err + "\nPlease use file opening version of mesh loading." );
        if ( SceneLoad::getSceneLoader( ext ) )
            return unexpected( err + "\nPlease open this format using scene loading function." );
        return unexpected( err );
    }

    auto res = loader.streamLoad( in, settings );
    if ( res )
        telemetryOpenMesh( ext, *res, settings );
    return res;
}

/*
MeshLoaderAdder __meshLoaderAdder( NamedMeshLoader{IOFilter( "MrMesh (.mrmesh)", "*.mrmesh" ),MeshLoader{static_cast<Expected<MR::Mesh>(*)(const std::filesystem::path&,VertColors*)>(fromMrmesh)}} );
*/

MR_ADD_MESH_LOADER_WITH_PRIORITY( IOFilter( "MeshInspector (.mrmesh)", "*.mrmesh" ), fromMrmesh, -1 )
MR_ADD_MESH_LOADER( IOFilter( "Stereolithography (.stl)", "*.stl" ), fromAnyStl )
MR_ADD_MESH_LOADER( IOFilter( "Object format file (.off)", "*.off" ), fromOff )
MR_ADD_MESH_LOADER( IOFilter( "3D model object (.obj)", "*.obj" ), fromObj )
MR_ADD_MESH_LOADER( IOFilter( "Polygon File Format (.ply)", "*.ply" ), fromPly )
MR_ADD_MESH_LOADER( IOFilter( "Drawing Interchange Format (.dxf)", "*.dxf" ), fromDxf )

} //namespace MeshLoad

} //namespace MR
