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

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, callback ), file );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    std::vector<NamedMesh> res;
    std::string currentObjName;
    std::vector<Vector3f> points;
    Triangulation t;

    auto finishObject = [&]() 
    {
        if ( !t.empty() )
        {
            res.emplace_back();
            res.back().name = std::move( currentObjName );

            // copy only minimal span of vertices for this object
            VertId minV(INT_MAX), maxV(-1);
            for ( const auto & vs : t )
            {
                minV = std::min( { minV, vs[0], vs[1], vs[2] } );
                maxV = std::max( { maxV, vs[0], vs[1], vs[2] } );
            }
            for ( auto & vs : t )
            {
                for ( int i = 0; i < 3; ++i )
                    vs[i] -= minV;
            }

            res.back().mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                VertCoords( points.begin() + minV, points.begin() + maxV + 1 ), t );
            t.clear();
        }
        currentObjName.clear();
    };

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    std::vector<int> vs;
    for ( int i = 0;; ++i )
    {
        if ( !in )
            return tl::make_unexpected( std::string( "OBJ-format read error" ) );
        char ch = 0;
        in >> ch;
        if ( in.eof() )
            break;
        if ( ch == 'v' && in.peek() != 'n' /*normals*/ && in.peek() != 't' /*texture coordinates*/ )
        {
            float x, y, z;
            in >> x >> y >> z;
            points.emplace_back( x, y, z );
        }
        else if ( ch == 'f' )
        {
            auto skipSlashNum = [&]()
            {
                auto s = (char)in.peek();
                if ( s != '/' )
                    return false;
                (void)in.get();
                auto s1 = (char)in.peek();
                if ( s1 >= '0' && s1 <= '9' )
                {
                    int x; //just skip for now
                    in >> x;
                }
                return true;
            };
            auto readVert = [&]()
            {
                int v;
                in >> v;
                if ( skipSlashNum() )
                    skipSlashNum();
                return v;
            };

            vs.clear();
            for ( int v = readVert(); in; v = readVert() )
                vs.push_back( v );
            if ( vs.size() < 3 )
                return tl::make_unexpected( std::string( "Face with less than 3 vertices in OBJ-file" ) );
            if ( !in.bad() && !in.eof() )
                in.clear();
            
            // TODO: make smarter triangulation based on point coordinates
            for ( int j = 1; j + 1 < vs.size(); ++j )
                t.push_back( { VertId( vs[0]-1 ), VertId( vs[j]-1 ), VertId( vs[j+1]-1 ) } );
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
        if ( callback && !(i & 0x3FF) )
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
