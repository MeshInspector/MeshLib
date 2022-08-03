#include "MRMeshLoadObj.h"
#include "MRStringConvert.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"

namespace MR
{

namespace MeshLoad
{

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    std::ifstream in( file );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromSceneObjFile( in, combineAllObjects, callback );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    std::vector<NamedMesh> res;
    std::string currentObjName;
    std::vector<Vector3f> points;
    std::vector<MeshBuilder::Triangle> tris;

    auto finishObject = [&]() 
    {
        if ( !tris.empty() )
        {
            res.emplace_back();
            res.back().name = std::move( currentObjName );

            // copy only minimal span of vertices for this object
            VertId minV(INT_MAX), maxV(-1);
            for ( const auto & t : tris )
            {
                minV = std::min( { minV, t.v[0], t.v[1], t.v[2] } );
                maxV = std::max( { maxV, t.v[0], t.v[1], t.v[2] } );
            }
            for ( auto & t : tris )
            {
                for ( int i = 0; i < 3; ++i )
                    t.v[i] -= minV;
            }

            res.back().mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                VertCoords( points.begin() + minV, points.begin() + maxV + 1 ), tris );
            tris.clear();
        }
        currentObjName.clear();
    };

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    for ( int i = 0;; ++i )
    {
        if ( !in )
            return tl::make_unexpected( std::string( "OBJ-format read error" ) );
        char ch = 0;
        in >> ch;
        if ( in.eof() )
            break;
        if ( ch == 'v' && in.peek() != 'n' )
        {
            float x, y, z;
            in >> x >> y >> z;
            points.emplace_back( x, y, z );
        }
        else if ( ch == 'f' )
        {
            auto readVert = [&]()
            {
                int v;
                in >> v;
                auto s = (char)in.peek();
                if ( s == '/' )
                {
                    (void)in.get();
                    auto s1 = (char)in.peek();
                    if ( s1 == '/' )
                    {
                        (void)in.get();
                        int x; //just skip for now
                        in >> x;
                    }
                }
                return v;
            };

            int a = readVert();
            int b = readVert();
            int c = readVert();
            tris.emplace_back( VertId( a-1 ), VertId( b-1 ), VertId( c-1 ), FaceId( tris.size() ) );
        }
        else if ( ch == 'o' )
        {
            if ( !combineAllObjects )
                finishObject();
            // next object
            getline( in, currentObjName );
            while ( !currentObjName.empty() && currentObjName[0] == ' ' )
                currentObjName.erase( currentObjName.begin() );
            {

            }
        }
        else
        {
            // skip unknown line
            std::string str;
            std::getline( in, str );
        }
        if ( callback && !(i % 1000) )
        {
            const float progress = int( in.tellg() - posStart ) / streamSize;
            if ( !callback( progress ) )
                return tl::make_unexpected( std::string( "Loading canceled" ));
        }
        if ( in.eof() )
            break;
    }

    finishObject();
    return res;
}

} //namespace MeshLoad

} //namespace MR
