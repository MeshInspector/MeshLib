#include "MRMeshLoad.h"
#include "MRMeshBuilder.h"
#include "MRMesh.h"
#include "MRphmap.h"
#include "MRTimer.h"
#include "miniply.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"
#include "MRMeshLoadObj.h"
#include "MRColor.h"
#include "OpenCTM/openctm.h"
#include <tbb/parallel_for.h>
#include <array>
#include <future>

namespace MR
{

namespace MeshLoad
{

tl::expected<Mesh, std::string> fromMrmesh( const std::filesystem::path & file, std::vector<Color>* )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromMrmesh( in );
}

tl::expected<Mesh, std::string> fromMrmesh( std::istream & in )
{
    MR_TIMER

    Mesh mesh;
    if ( !mesh.topology.read( in ) )
        return tl::make_unexpected( std::string( "Error reading topology from mrmesh-file" ) );

    // read points
    std::uint32_t numPoints;
    in.read( (char*)&numPoints, 4 );
    if ( !in )
        return tl::make_unexpected( std::string( "Error reading the number of points from mrmesh-file" ) );
    mesh.points.resize( numPoints );
    in.read( (char*)mesh.points.data(), mesh.points.size() * sizeof(Vector3f) );
    if ( !in )
        return tl::make_unexpected( std::string( "Error reading  points from mrmesh-file" ) );

    return std::move( mesh );
}

tl::expected<Mesh, std::string> fromOff( const std::filesystem::path & file, std::vector<Color>* )
{
    std::ifstream in( file );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromOff( in );
}

tl::expected<Mesh, std::string> fromOff( std::istream & in )
{
    MR_TIMER
    std::string header;
    in >> header;
    if ( !in || header != "OFF" )
        return tl::make_unexpected( std::string( "File is not in OFF-format" ) );

    int numPoints, numPolygons, numUnused;
    in >> numPoints >> numPolygons >> numUnused;
    if ( !in || numPoints <= 0 || numPolygons <= 0 || numUnused != 0 )
        return tl::make_unexpected( std::string( "Unsupported OFF-format" ) );

    std::vector<Vector3f> points;
    points.reserve( numPoints );

    for ( int i = 0; i < numPoints; ++i )
    {
        float x, y, z;
        in >> x >> y >> z;
        if ( !in )
            return tl::make_unexpected( std::string( "Points read error" ) );
        points.emplace_back( x, y, z );
    }

    std::vector<MeshBuilder::Triangle> tris;
    tris.reserve( numPolygons );

    for ( int i = 0; i < numPolygons; ++i )
    {
        int k, a, b, c;
        in >> k >> a >> b >> c;
        if ( !in || k != 3 )
            return tl::make_unexpected( std::string( "Polygons read error" ) );
        tris.emplace_back( VertId( a ), VertId( b ), VertId( c ), FaceId( i ) );
    }

    Mesh res;
    res.topology = MeshBuilder::fromTriangles( tris );
    res.points.vec_ = std::move( points );

    return std::move( res );
}

tl::expected<Mesh, std::string> fromObj( const std::filesystem::path & file, std::vector<Color>* )
{
    std::ifstream in( file );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromObj( in );
}

tl::expected<Mesh, std::string> fromObj( std::istream & in )
{
    MR_TIMER

    auto objs = fromSceneObjFile( in, true );
    if ( !objs.has_value() )
        return tl::make_unexpected( objs.error() );
    if ( objs->size() != 1 )
        return tl::make_unexpected( "OBJ-file is empty" );

    return std::move( (*objs)[0].mesh );
}

tl::expected<MR::Mesh, std::string> fromAnyStl( const std::filesystem::path& file, std::vector<Color>* )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromAnyStl( in );
}

tl::expected<MR::Mesh, std::string> fromAnyStl( std::istream& in )
{
    auto pos = in.tellg();
    auto resBin = fromBinaryStl( in );
    if ( resBin.has_value() )
        return resBin;
    in.clear();
    in.seekg( pos );
    auto resAsc = fromASCIIStl( in );
    if ( resAsc.has_value() )
        return resAsc;
    return tl::make_unexpected( resBin.error() + '\n' + resAsc.error() );
}

tl::expected<Mesh, std::string> fromBinaryStl( const std::filesystem::path & file, std::vector<Color>* )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromBinaryStl( in );
}

// this makes bit-wise comparison of two Vector3f's thus making two NaNs equal
struct equalVector3f
{
    bool operator() ( const Vector3f & a, const Vector3f & b ) const
    {
        static_assert( sizeof( Vector3f ) == 12 );
        char ax[12], bx[12];
        memcpy( ax, &a, 12 );
        memcpy( bx, &b, 12 );
        return memcmp( ax, bx, 12 ) == 0;
    }
};

tl::expected<Mesh, std::string> fromBinaryStl( std::istream & in )
{
    MR_TIMER

    char header[80];
    in.read( header, 80 );

    std::uint32_t numTris;
    in.read( (char*)&numTris, 4 );
    if ( !in )
        return tl::make_unexpected( std::string( "Error reading the number of triangles from STL-file" ) );

    auto posCur = in.tellg();
    in.seekg( 0, std::ios_base::end );
    auto posEnd = in.tellg();
    in.seekg( posCur );
    if ( posEnd - posCur < 50 * numTris )
        return tl::make_unexpected( std::string( "Binary STL-file is too short" ) );

    using HMap = phmap::parallel_flat_hash_map<Vector3f, VertId, phmap::priv::hash_default_hash<Vector3f>, equalVector3f>;

    HMap hmap;
    hmap.reserve( numTris / 2 ); // there should be about twice more triangles than vertices
    Mesh res;

    std::vector<MeshBuilder::Triangle> tris;
    tris.reserve( numTris );

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

    // first chunk
    in.read( (char*)buffer.data(), sizeof(StlTriangle) * itemsInBuffer );
    if ( !in  )
        return tl::make_unexpected( std::string( "Binary STL read error" ) );

    using VertInHMap = std::array<VertId*, 3>;
    std::vector<VertInHMap> vertsInHMap( itemsInBuffer );

    for ( std::uint32_t i = 0;; )
    {
        std::future<void> f;
        if ( i + buffer.size() < numTris )
        {
            const auto itemsInNextChuck = std::min( numTris - i - (std::uint32_t)buffer.size(), itemsInBuffer );
            nextBuffer.resize( itemsInNextChuck );
            f = std::async( std::launch::async, [&in, &nextBuffer]()
            {
                in.read( (char*)nextBuffer.data(), sizeof(StlTriangle) * nextBuffer.size() );
            } );
        }

        for (;;)
        {
            auto buckets0 = hmap.bucket_count();

            const auto subcnt = hmap.subcnt();
            tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ), [&]( const tbb::blocked_range<size_t> & range )
            {
                assert( range.begin() + 1 == range.end() );
                for ( size_t myPartId = range.begin(); myPartId < range.end(); ++myPartId )
                {
                    for ( size_t j = 0; j < buffer.size(); ++j )
                    {
                        const auto & st = buffer[j];
                        auto & it = vertsInHMap[j];
                        for ( int k = 0; k < 3; ++k )
                        {
                            const auto & p = st.vert[k];
                            auto hashval = hmap.hash( p );
                            auto idx = hmap.subidx( hashval );
                            if ( idx != myPartId )
                                continue;
                            it[k] = &hmap[ p ];
                        }
                    }
                }
            } );

            if ( buckets0 == hmap.bucket_count() )
                break; // the number of buckets has not changed - all pointers are valid
        }

        for ( size_t j = 0; j < buffer.size(); ++j )
        {
            const auto & st = buffer[j];
            const auto & it = vertsInHMap[j];
            for ( int k = 0; k < 3; ++k )
            {
                if ( !it[k]->valid() )
                {
                    *it[k] = VertId( (int)res.points.size() );
                    res.points.push_back( st.vert[k] );
                }
            }
            tris.emplace_back( *it[0], *it[1], *it[2], FaceId( int( i++ ) ) );
        }

        if ( !f.valid() )
            break;
        f.get();
        if ( !in  )
            return tl::make_unexpected( std::string( "Binary STL read error" ) );
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

    res.topology = MeshBuilder::fromTriangles( tris );

    return std::move( res );
}

tl::expected<Mesh, std::string> fromASCIIStl( const std::filesystem::path& file, std::vector<Color>* )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromASCIIStl( in );

}

tl::expected<Mesh, std::string> fromASCIIStl( std::istream& in )
{
    MR_TIMER;

    using HMap = ParallelHashMap<Vector3f, VertId>;
    HMap hmap;
    Mesh res;
    std::vector<MeshBuilder::Triangle> tris;

    std::string line;
    std::string prefix;
    Vector3f point;
    int f{0};
    MeshBuilder::Triangle currTri;
    int triPos = 0;
    bool solidFound = false;
    while ( std::getline( in, line ) )
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
                id = VertId( res.points.size() );
                res.points.push_back( point );
            }
            currTri.v[triPos] = id;
            ++triPos;
            continue;
        }
        if ( prefix == "endloop" )
        {
            currTri.f = FaceId( f );
            tris.push_back( currTri );
            ++f;
            continue;
        }
    }

    if ( !solidFound )
        return tl::make_unexpected( std::string( "Failed to find 'solid' prefix in ascii STL" ) );

    res.topology = MeshBuilder::fromTriangles( tris );

    return std::move( res );
}

tl::expected<Mesh, std::string> fromPly( const std::filesystem::path& file, std::vector<Color>* colors )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromPly( in, colors );
}

tl::expected<Mesh, std::string> fromPly( std::istream& in, std::vector<Color>* colors )
{
    MR_TIMER

    miniply::PLYReader reader( in );
    if ( !reader.valid() )
        return tl::make_unexpected( std::string( "PLY file open error" ) );

    uint32_t indecies[3];
    bool gotVerts = false, gotFaces = false;

    std::vector<unsigned char> colorsBuffer;
    Mesh res;
    for ( ; reader.has_element() && ( !gotVerts || !gotFaces ); reader.next_element() ) 
    {
        if ( reader.element_is(miniply::kPLYVertexElement) && reader.load_element()  ) 
        {
            auto numVerts = reader.num_rows();
            if ( reader.find_pos( indecies ) )
            {
                res.points.resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, res.points.data() );
                gotVerts = true;
            }
            if ( colors && reader.find_color( indecies ) )
            {
                colorsBuffer.resize( 3 * numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
            }
            continue;
        }
        if ( reader.element_is(miniply::kPLYFaceElement) && reader.load_element() && reader.find_indices(indecies) )
        {
            bool polys = reader.requires_triangulation( indecies[0] );
            if ( polys && !gotVerts )
                return tl::make_unexpected( std::string( "PLY file open: need vertex positions to triangulate faces" ) );

            std::vector<VertId> vertTriples;
            if (polys) 
            {
                auto numIndices = reader.num_triangles( indecies[0] ) * 3;
                vertTriples.resize( numIndices );
                reader.extract_triangles( indecies[0], &res.points.front().x, (std::uint32_t)res.points.size(), miniply::PLYPropertyType::Int, &vertTriples.front() );
            }
            else 
            {
                auto numIndices = reader.num_rows() * 3;
                vertTriples.resize( numIndices );
                reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Int, &vertTriples.front() );
            }
            res.topology = MeshBuilder::fromVertexTriples( vertTriples );
            gotFaces = true;
        }
    }

    if ( !reader.valid() )
        return tl::make_unexpected( std::string( "PLY file read or parse error" ) );

    if ( !gotVerts )
        return tl::make_unexpected( std::string( "PLY file does not contain vertices" ) );

    if ( !gotFaces )
        return tl::make_unexpected( std::string( "PLY file does not contain faces" ) );

    if ( colors && !colorsBuffer.empty() )
    {
        colors->resize( res.points.size() );
        for ( int i = 0; i < res.points.size(); ++i )
        {
            int ind = 3 * i;
            ( *colors )[i] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
        }
    }


    return std::move( res );
}

tl::expected<Mesh, std::string> fromCtm( const std::filesystem::path & file, std::vector<Color>* colors )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromCtm( in, colors );
}

tl::expected<Mesh, std::string> fromCtm( std::istream & in, std::vector<Color>* colors )
{
    MR_TIMER

    class ScopedCtmConext 
    {
        CTMcontext context_ = ctmNewContext( CTM_IMPORT );
    public:
        ~ScopedCtmConext() { ctmFreeContext( context_ ); }
        operator CTMcontext() { return context_; }
    } context;

    ctmLoadCustom( context, []( void * buf, CTMuint size, void * data )
    {
        std::istream & s = *reinterpret_cast<std::istream *>( data );
        auto pos = s.tellg();
        s.read( (char*)buf, size );
        return (CTMuint)( s.tellg() - pos );
    }, &in );

    auto vertCount = ctmGetInteger( context, CTM_VERTEX_COUNT );
    auto triCount  = ctmGetInteger( context, CTM_TRIANGLE_COUNT );
    auto vertices  = ctmGetFloatArray( context, CTM_VERTICES );
    auto indices   = ctmGetIntegerArray( context, CTM_INDICES );
    if ( ctmGetError(context) != CTM_NONE )
        return tl::make_unexpected( "Error reading CTM format" );

    if ( triCount == 1 &&
         indices[0] == 0 && indices[1] == 0 && indices[2] == 0 )
        return tl::make_unexpected( "CTM File is representing points" );

    if ( colors )
    {
        auto colorAttrib = ctmGetNamedAttribMap( context, "Color" );
        if ( colorAttrib != CTM_NONE )
        {
            auto colorArray = ctmGetFloatArray( context, colorAttrib );
            colors->resize( vertCount );
            for ( CTMuint i = 0; i < vertCount; ++i )
            {
                auto j = 4 * i;
                (*colors)[i] = Color( colorArray[j], colorArray[j + 1], colorArray[j + 2], colorArray[j + 3] );
            }
        }
    }

    Mesh mesh;
    mesh.points.resize( vertCount );
    for ( VertId i{0}; i < (int)vertCount; ++i )
        mesh.points[i] = Vector3f( vertices[3*i], vertices[3*i+1], vertices[3*i+2] );

    std::vector<MeshBuilder::Triangle> tris;
    tris.reserve( triCount );
    for ( FaceId i{0}; i < (int)triCount; ++i )
        tris.emplace_back( VertId( (int)indices[3*i] ), VertId( (int)indices[3*i+1] ), VertId( (int)indices[3*i+2] ), i );

    mesh.topology = MeshBuilder::fromTriangles( tris );

    return std::move( mesh );
}

tl::expected<Mesh, std::string> fromAnySupportedFormat( const std::filesystem::path & file, std::vector<Color>* colors )
{
    auto ext = file.extension().u8string();
    for ( auto & c : ext )
        c = (char)tolower( c );

    ext = u8"*" + ext;

    tl::expected<MR::Mesh, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    auto filters = getFilters();
    auto itF = std::find_if( filters.begin(), filters.end(), [ext]( const IOFilter& filter )
    {
        return filter.extension == asString( ext );
    } );
    if ( itF == filters.end() )
        return res;

    auto loader = getMeshLoader( *itF );
    if ( !loader )
        return res;
    return loader( file, colors );
}

/*
MeshLoaderAdder __meshLoaderAdder( NamedMeshLoader{IOFilter( "MrMesh (.mrmesh)", "*.mrmesh" ),MeshLoader{static_cast<tl::expected<MR::Mesh, std::string>(*)(const std::filesystem::path&,std::vector<Color>*)>(fromMrmesh)}} );
*/

MR_ADD_MESH_LOADER( IOFilter( "MeshRUs Mesh (.mrmesh)", "*.mrmesh" ), fromMrmesh )
MR_ADD_MESH_LOADER( IOFilter( "Stereolithography (.stl)", "*.stl" ), fromAnyStl )
MR_ADD_MESH_LOADER( IOFilter( "Object format file (.off)", "*.off" ), fromOff )
MR_ADD_MESH_LOADER( IOFilter( "3D model object (.obj)", "*.obj" ), fromObj )
MR_ADD_MESH_LOADER( IOFilter( "Polygon File Format (.ply)", "*.ply" ), fromPly )
MR_ADD_MESH_LOADER( IOFilter( "Compact triangle-based mesh (.ctm)", "*.ctm" ), fromCtm )

} //namespace MeshLoad

} //namespace MR
