#include "MRObjectLoad.h"
#include "MRObjectMesh.h"
#include "MRMeshLoad.h"
#include "MRLinesLoad.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRDistanceMapLoad.h"
#include "MRImageLoad.h"
#include "MRPointsLoad.h"
#include "MRObjectFactory.h"
#include "MRObjectLines.h"
#include "MRObjectPoints.h"
#include "MRDistanceMap.h"
#include "MRObjectDistanceMap.h"
#include "MRObjectGcode.h"
#include "MRPointCloud.h"
#include "MRGcodeLoad.h"
#include "MRStringConvert.h"
#include "MRIOFormatsRegistry.h"
#include "MRMeshLoadObj.h"
#include "MRSerializer.h"
#include "MRDirectory.h"
#include "MRSceneSettings.h"
#include "MRMeshLoadSettings.h"
#include "MRZip.h"
#include "MRVoxels/MRDicom.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace
{

/// finds if given mesh has enough sharp edges (>25 degrees) to recommend flat shading
bool detectFlatShading( const Mesh& mesh )
{
    MR_TIMER

    constexpr float sharpAngle = 25 * PI_F / 180; // Critical angle from planar, degrees
    const float sharpAngleCos = std::cos( sharpAngle );

    struct Data
    {
        double sumDblArea = 0;
        double sumSharpDblArea = 0;
        Data operator + ( const Data & b ) const 
        {
            return { sumDblArea + b.sumDblArea, sumSharpDblArea + b.sumSharpDblArea };
        }
    };

    auto total = parallel_deterministic_reduce(
        tbb::blocked_range( 0_ue, UndirectedEdgeId{ mesh.topology.undirectedEdgeSize() } ),
        Data(),
        [&mesh, sharpAngleCos] ( const auto& range, Data current )
        {
            for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
            {
                const EdgeId e = ue;
                const auto l = mesh.topology.left( e );
                const auto r = mesh.topology.right( e );
                if ( !l || !r )
                    continue;
                const auto da = mesh.dblArea( l ) + mesh.dblArea( r );
                current.sumDblArea += da;
                auto dihedralCos = mesh.dihedralAngleCos( ue );
                if ( dihedralCos <= sharpAngleCos )
                    current.sumSharpDblArea += da;
            }
            return current;
        },
        std::plus<Data>() );

    // triangles' area near sharp edges is more than 5% of total area
    return total.sumSharpDblArea > 0.05 * total.sumDblArea;
}

// Prepare object after it has been imported from external format (not .mru)
void postImportObject( const std::shared_ptr<Object> &o, const std::filesystem::path &filename )
{
    if ( std::shared_ptr<ObjectMesh> mesh = std::dynamic_pointer_cast< ObjectMesh >( o ) )
    {
        // Detect flat shading needed
        bool flat;
        if ( SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::AutoDetect )
            flat = filename.extension() == ".step" || filename.extension() == ".stp" ||
                   ( mesh->mesh() && detectFlatShading( *mesh->mesh().get() ) );
        else
            flat = SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::Flat;
        mesh->setVisualizeProperty( flat, MeshVisualizePropertyType::FlatShading, ViewportMask::all() );
    }
    for ( const std::shared_ptr<Object>& child : o->children() )
        postImportObject( child, filename );
}

} // anonymous namespace

IOFilters getAllFilters()
{
    return
        SceneLoad::getFilters()
        | ObjectLoad::getFilters()
        | MeshLoad::getFilters()
        | LinesLoad::getFilters()
        | PointsLoad::getFilters()
    ;
}

Expected<ObjectMesh> makeObjectMeshFromFile( const std::filesystem::path& file, const MeshLoadInfo& info /*= {}*/ )
{
    auto expObj = makeObjectFromMeshFile( file, info, true );
    if ( !expObj )
        return unexpected( std::move( expObj.error() ) );

    auto * mesh = dynamic_cast<ObjectMesh*>( expObj.value().get() );
    if ( !mesh )
    {
        assert( false );
        return unexpected( "makeObjectFromMeshFile returned not a mesh" );
    }

    return std::move( *mesh );
}

static std::string makeWarningString( int skippedFaceCount, int duplicatedVertexCount, int holesCount )
{
    std::string res;
    if ( skippedFaceCount )
        res = fmt::format( "{} triangles were skipped as inconsistent with others.", skippedFaceCount );
    if ( duplicatedVertexCount )
    {
        if ( !res.empty() )
            res += '\n';
        res += fmt::format( "{} vertices were duplicated to make them manifold.", duplicatedVertexCount );
    }
    if ( holesCount )
    {
        if ( !res.empty() )
            res += '\n';
        res += fmt::format( "The objects contains {} holes. Please consider using Fill Holes tool.", holesCount );
    }
    return res;
}

Expected<std::shared_ptr<Object>> makeObjectFromMeshFile( const std::filesystem::path& file, const MeshLoadInfo& info, bool returnOnlyMesh )
{
    MR_TIMER

    VertColors colors;
    VertUVCoords uvCoords;
    VertNormals normals;
    MeshTexture texture;
    int skippedFaceCount = 0;
    int duplicatedVertexCount = 0;
    int holesCount = 0;
    AffineXf3f xf;
    MeshLoadSettings settings
    {
        .colors = &colors,
        .uvCoords = &uvCoords,
        .normals = returnOnlyMesh ? nullptr : &normals,
        .texture = &texture,
        .skippedFaceCount = info.warnings ? &skippedFaceCount : nullptr,
        .duplicatedVertexCount = info.warnings ? &duplicatedVertexCount : nullptr,
        .xf = &xf,
        .callback = info.callback
    };
    auto mesh = MeshLoad::fromAnySupportedFormat( file, settings );
    if ( !mesh.has_value() )
        return unexpected( mesh.error() );
    
    if ( !mesh->points.empty() && mesh->topology.numValidFaces() <= 0 )
    {
        if ( returnOnlyMesh )
            return unexpected( "File contains a point cloud and not a mesh: " + utf8string( file ) );
        auto pointCloud = std::make_shared<MR::PointCloud>();
        pointCloud->points = std::move( mesh->points );
        pointCloud->normals = std::move( normals );
        pointCloud->validPoints.resize( pointCloud->points.size(), true );

        auto objectPoints = std::make_unique<ObjectPoints>();
        objectPoints->setName( utf8string( file.stem() ) );
        objectPoints->setPointCloud( pointCloud );

        if ( !colors.empty() )
        {
            objectPoints->setVertsColorMap( std::move( colors ) );
            objectPoints->setColoringType( ColoringType::VertsColorMap );
        }

        objectPoints->setXf( xf );

        return objectPoints;
    }

    const auto numVerts = mesh->points.size();
    const bool hasColors = colors.size() >= numVerts;
    const bool hasUV = uvCoords.size() >= numVerts;
    const bool hasTexture = !texture.pixels.empty();

    auto objectMesh = std::make_unique<ObjectMesh>();
    objectMesh->setName( utf8string( file.stem() ) );
    objectMesh->setMesh( std::make_shared<Mesh>( std::move( mesh.value() ) ) );

    if ( hasColors )
        objectMesh->setVertsColorMap( std::move( colors ) );
    if ( hasUV )
        objectMesh->setUVCoords( std::move( uvCoords ) );
    if ( hasTexture )
        objectMesh->setTextures( { std::move( texture ) } );

    if ( hasUV && hasTexture )
        objectMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
    else if ( hasColors )
        objectMesh->setColoringType( ColoringType::VertsColorMap );

    objectMesh->setXf( xf );
    if ( info.warnings )
    {
        holesCount = int( objectMesh->numHoles() );
        if ( !info.warnings->empty() )
            *info.warnings += '\n';
        auto s = makeWarningString( skippedFaceCount, duplicatedVertexCount, holesCount );
        if ( !s.empty() )
        {
            *info.warnings += s;
            *info.warnings += '\n';
        }
        if ( !colors.empty() && !hasColors )
            *info.warnings += fmt::format( "Ignoring too few ({}) colors loaded for a mesh with {} vertices.\n", colors.size(), numVerts );
        if ( !uvCoords.empty() && !hasUV )
            *info.warnings += fmt::format( "Ignoring too few ({}) uv-coordinates loaded for a mesh with {} vertices.\n", uvCoords.size(), numVerts );
        if ( !info.warnings->empty() && info.warnings->back() == '\n' )
            info.warnings->pop_back();
    }

    return objectMesh;
}

Expected<ObjectLines> makeObjectLinesFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    auto lines = LinesLoad::fromAnySupportedFormat( file, callback );
    if ( !lines.has_value() )
    {
        return unexpected( lines.error() );
    }

    ObjectLines objectLines;
    objectLines.setName( utf8string( file.stem() ) );
    objectLines.setPolyline( std::make_shared<MR::Polyline3>( std::move( lines.value() ) ) );

    return objectLines;
}

Expected<ObjectPoints> makeObjectPointsFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    VertColors colors;
    AffineXf3f xf;
    auto pointsCloud = PointsLoad::fromAnySupportedFormat( file, {
        .colors = &colors,
        .outXf = &xf,
        .callback = callback,
    } );
    if ( !pointsCloud.has_value() )
    {
        return unexpected( pointsCloud.error() );
    }

    ObjectPoints objectPoints;
    objectPoints.setName( utf8string( file.stem() ) );
    objectPoints.setPointCloud( std::make_shared<MR::PointCloud>( std::move( pointsCloud.value() ) ) );
    objectPoints.setXf( xf );
    if ( !colors.empty() )
    {
        objectPoints.setVertsColorMap( std::move( colors ) );
        objectPoints.setColoringType( ColoringType::VertsColorMap );
    }

    return objectPoints;
}

Expected<ObjectDistanceMap> makeObjectDistanceMapFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    DistanceMapToWorld params;
    auto distanceMap = DistanceMapLoad::fromAnySupportedFormat( file, &params, callback );
    if ( !distanceMap.has_value() )
    {
        return unexpected( distanceMap.error() );
    }

    ObjectDistanceMap objectDistanceMap;
    objectDistanceMap.setName( utf8string( file.stem() ) );
    objectDistanceMap.setDistanceMap( std::make_shared<MR::DistanceMap>( std::move( distanceMap.value() ) ), params );

    return objectDistanceMap;
}

Expected<ObjectGcode> makeObjectGcodeFromFile( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;

    auto gcodeSource = GcodeLoad::fromAnySupportedFormat( file, callback );
    if ( !gcodeSource.has_value() )
        return unexpected( std::move( gcodeSource.error() ) );

    ObjectGcode objectGcode;
    objectGcode.setName( utf8string( file.stem() ) );
    objectGcode.setGcodeSource( std::make_shared<GcodeSource>( std::move( *gcodeSource ) ) );

    return objectGcode;
}

Expected<std::vector<std::shared_ptr<MR::Object>>> loadObjectFromFile( const std::filesystem::path& filename,
                                                                       std::string* loadWarn, ProgressCallback callback )
{
    if ( callback && !callback( 0.f ) )
        return unexpected( std::string( "Loading canceled" ) );

    Expected<std::vector<std::shared_ptr<Object>>> result;
    bool loadedFromSceneFile = false;

    auto ext = std::string( "*" ) + utf8string( filename.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );   
    
    if ( findFilter( SceneLoad::getFilters(), ext ) )
    {
        const auto objTree = loadSceneFromAnySupportedFormat( filename, loadWarn, callback );
        if ( !objTree.has_value() )
            return unexpected( objTree.error() );
        
        result = std::vector( { *objTree } );
        ( *result )[0]->setName( utf8string( filename.stem() ) );
        loadedFromSceneFile = true;
    }
    else if ( const auto filter = findFilter( ObjectLoad::getFilters(), ext ) )
    {
        const auto loader = ObjectLoad::getObjectLoader( *filter );
        result = loader( filename, loadWarn, std::move( callback ) );
    }
    else
    {
        MeshLoadInfo info
        {
            .warnings = loadWarn,
            .callback = callback
        };
        auto object = makeObjectFromMeshFile( filename, info );
        if ( object && *object )
        {
            (*object)->select( true );
            result = { *object };
        }
        else if ( object.error() == "Loading canceled" )
        {
            result = unexpected( std::move( object.error() ) );
        }
        else
        {
            result = unexpected( std::move( object.error() ) );

            auto objectPoints = makeObjectPointsFromFile( filename, callback );
            if ( objectPoints.has_value() )
            {
                objectPoints->select( true );
                auto obj = std::make_shared<ObjectPoints>( std::move( objectPoints.value() ) );
                result = { obj };
            }
            else if ( result.error() == "unsupported file extension" )
            {
                result = unexpected( objectPoints.error() );

                auto objectLines = makeObjectLinesFromFile( filename, callback );
                if ( objectLines.has_value() )
                {
                    objectLines->select( true );
                    auto obj = std::make_shared<ObjectLines>( std::move( objectLines.value() ) );
                    result = { obj };
                }
                else if ( result.error() == "unsupported file extension" )
                {
                    result = unexpected( objectLines.error() );

                    auto objectDistanceMap = makeObjectDistanceMapFromFile( filename, callback );
                    if ( objectDistanceMap.has_value() )
                    {
                        objectDistanceMap->select( true );
                        auto obj = std::make_shared<ObjectDistanceMap>( std::move( objectDistanceMap.value() ) );
                        result = { obj };
                    }
                    else if ( result.error() == "unsupported file extension" )
                    {
                        result = unexpected( objectDistanceMap.error() );

                        auto objectGcode = makeObjectGcodeFromFile( filename, callback );
                        if ( objectGcode.has_value() )
                        {
                            objectGcode->select( true );
                            auto obj = std::make_shared<ObjectGcode>( std::move( objectGcode.value() ) );
                            result = { obj };
                        }
                        else
                        {
                            result = unexpected( objectGcode.error() );
                        }
                    }
                }
            }
        }
    }

    if ( result.has_value() && !loadedFromSceneFile )
        for ( const std::shared_ptr<Object>& o : result.value() )
        {
            postImportObject( o, filename );
            if ( auto objectPoints = o->asType<ObjectPoints>(); objectPoints && loadWarn )
            {
                if ( !objectPoints->pointCloud()->hasNormals() )
                    *loadWarn += "Point cloud " + o->name() + " has no normals.\n";
                if ( objectPoints->getRenderDiscretization() > 1 )
                    *loadWarn += "Point cloud " + o->name() + " has too many points in PointCloud:\n"
                    "Visualization is simplified (only part of the points is drawn)\n";
            }
        }

    if ( !result.has_value() )
        spdlog::error( result.error() );

    return result;
}


bool isSupportedFileInSubfolders( const std::filesystem::path& folder )
{
    const auto allFilters = getAllFilters();

    std::vector<std::filesystem::path> filesList;
    filesList.push_back( folder );

    while ( !filesList.empty() )
    {
        auto path = filesList[0];
        filesList.erase( filesList.begin() );

        std::error_code ec;
        for ( auto entry : Directory{ path, ec } )
        {
            auto subpath = entry.path();
            if ( entry.is_directory( ec ) )
            {
                filesList.push_back( path = subpath );
            }
            else if ( entry.is_regular_file( ec ) )
            {
                auto ext = utf8string( subpath.extension() );
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                if ( ext.empty() )
                    continue;

                if ( findFilter( allFilters, ext ) )
                    return true;
            }
        }
    }
    return false;
}

Expected<std::shared_ptr<Object>> loadSceneFromAnySupportedFormat( const std::filesystem::path& path, std::string* loadWarn,
    ProgressCallback callback )
{
    auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto loader = SceneLoad::getSceneLoader( ext );
    if ( !loader )
        return unexpected( std::string( "unsupported file extension" ) );

    return loader( path, loadWarn, callback )
    .and_then( [&] ( ObjectPtr&& obj ) -> Expected<ObjectPtr>
    {
        if ( ext != "*.mru" && ext != "*.zip" )
            postImportObject( obj, path );

        return std::move( obj );
    } );
}

Expected<std::shared_ptr<Object>> deserializeObjectTree( const std::filesystem::path& path, FolderCallback postDecompress,
                                                         ProgressCallback progressCb )
{
    MR_TIMER;
    UniqueTemporaryFolder scenePath( postDecompress );
    if ( !scenePath )
        return unexpected( "Cannot create temporary folder" );
    auto res = decompressZip( path, scenePath );
    if ( !res.has_value() )
        return unexpected( std::move( res.error() ) );

    return deserializeObjectTreeFromFolder( scenePath, progressCb );
}

Expected<std::shared_ptr<Object>> deserializeObjectTreeFromFolder( const std::filesystem::path& folder,
                                                                   ProgressCallback progressCb )
{
    MR_TIMER;

    std::error_code ec;
    std::filesystem::path jsonFile;
    for ( auto entry : Directory{ folder, ec } )
    {
        // unlike extension() this works even if full file name is simply ".json"
        if ( entry.path().u8string().ends_with( u8".json" ) )
        {
            jsonFile = entry.path();
            break;
        }
    }

    auto readRes = deserializeJsonValue( jsonFile );
    if( !readRes.has_value() )
    {
        return unexpected( readRes.error() );
    }
    auto root = readRes.value();

    auto typeTreeSize = root["Type"].size();
    std::shared_ptr<Object> rootObject;
    for (int i = typeTreeSize-1;i>=0;--i)
    {
        const auto& type = root["Type"][unsigned( i )];
        if ( type.isString() )
            rootObject = createObject( type.asString() );
        if ( rootObject )
            break;
    }
    if ( !rootObject )
        return unexpected( "Unknown root object type" );

    int modelNumber{ 0 };
    int modelCounter{ 0 };
    if ( progressCb )
    {
        std::function<int( const Json::Value& )> calculateModelNum = [&calculateModelNum] ( const Json::Value& root )
        {
            int res{ 1 };

            if ( root["Children"].isNull() )
                return res;

            for ( const std::string& childKey : root["Children"].getMemberNames() )
            {
                if ( !root["Children"].isMember( childKey ) )
                    continue;

                const auto& child = root["Children"][childKey];
                if ( child.isNull() )
                    continue;
                res += calculateModelNum( child );
            }

            return res;
        };
        modelNumber = calculateModelNum( root );

        modelNumber = std::max( modelNumber, 1 );
        progressCb = [progressCb, &modelCounter, modelNumber] ( float v )
        {
            return progressCb( ( modelCounter + v ) / modelNumber );
        };
    }

    auto resDeser = rootObject->deserializeRecursive( folder, root, progressCb, &modelCounter );
    if ( !resDeser.has_value() )
    {
        std::string errorStr = resDeser.error();
        if ( errorStr != "Loading canceled" )
            errorStr = "Cannot deserialize: " + errorStr;
        return unexpected( errorStr );
    }

    return rootObject;
}

Expected<ObjectPtr> deserializeObjectTree( const std::filesystem::path& path, std::string*, ProgressCallback progressCb )
{
    return deserializeObjectTree( path, FolderCallback{}, std::move( progressCb ) );
}

MR_ADD_SCENE_LOADER_WITH_PRIORITY( IOFilter( "MeshInspector scene (.mru)", "*.mru" ), deserializeObjectTree, -1 )

} //namespace MR
