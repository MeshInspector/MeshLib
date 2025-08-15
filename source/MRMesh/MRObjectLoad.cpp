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
#include "MRPch/MRTBB.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRJson.h"

namespace MR
{

namespace
{

/// finds if given mesh has enough sharp edges (>25 degrees) to recommend flat shading
bool detectFlatShading( const Mesh& mesh )
{
    MR_TIMER;

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
    const auto extension = utf8string( filename.extension() );

    // TODO: get format id from the format registry
    const auto sourceFormatId = !extension.empty() ? toLower( extension.substr( 1 ) ) : "unknown";
    const auto sourceFormatTag = fmt::format( ".source-format:{}", sourceFormatId );
    o->addTag( sourceFormatTag );

    if ( std::shared_ptr<ObjectMesh> mesh = std::dynamic_pointer_cast< ObjectMesh >( o ) )
    {
        // Detect flat shading needed
        bool flat;
        if ( SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::AutoDetect )
            flat = extension == ".step" || extension == ".stp" ||
                   ( mesh->mesh() && detectFlatShading( *mesh->mesh().get() ) );
        else
            flat = SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::Flat;
        mesh->setVisualizeProperty( flat, MeshVisualizePropertyType::FlatShading, ViewportMask::all() );
    }
    for ( const std::shared_ptr<Object>& child : o->children() )
        postImportObject( child, filename );
}

} // namespace

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

Expected<LoadedObjectMesh> makeObjectMeshFromFile( const std::filesystem::path& file, const ProgressCallback& cb )
{
    auto maybe = makeObjectFromMeshFile( file, cb, true );
    if ( !maybe )
        return unexpected( std::move( maybe.error() ) );

    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( maybe->obj );
    if ( !objMesh )
    {
        assert( false );
        return unexpected( "makeObjectFromMeshFile returned not a mesh" );
    }

    return LoadedObjectMesh{ .obj = std::move( objMesh ), .warnings = std::move( maybe->warnings ) };
}

static std::string makeWarningString( int skippedFaceCount, int duplicatedVertexCount, int holesCount )
{
    std::string res;
    if ( skippedFaceCount )
        res = fmt::format( "{} triangles were skipped as inconsistent with others.\n", skippedFaceCount );
    if ( duplicatedVertexCount )
        res += fmt::format( "{} vertices were duplicated to make them manifold.\n", duplicatedVertexCount );
    if ( holesCount )
        res += fmt::format( "The objects contains {} holes. Please consider using Fill Holes tool.\n", holesCount );
    return res;
}

Expected<LoadedObject> makeObjectFromMeshFile( const std::filesystem::path& file, const ProgressCallback& cb, bool returnOnlyMesh )
{
    MR_TIMER;

    VertColors colors;
    VertUVCoords uvCoords;
    VertNormals normals;
    MeshTexture texture;
    std::optional<Edges> edges;
    int skippedFaceCount = 0;
    int duplicatedVertexCount = 0;
    int holesCount = 0;
    AffineXf3f xf;
    MeshLoadSettings settings
    {
        .edges = &edges,
        .colors = &colors,
        .uvCoords = &uvCoords,
        .normals = returnOnlyMesh ? nullptr : &normals,
        .texture = &texture,
        .skippedFaceCount = &skippedFaceCount,
        .duplicatedVertexCount = &duplicatedVertexCount,
        .xf = &xf,
        .callback = cb
    };
    auto mesh = MeshLoad::fromAnySupportedFormat( file, settings );
    if ( !mesh.has_value() )
        return unexpected( mesh.error() );

    if ( !mesh->points.empty() && mesh->topology.numValidFaces() <= 0 )
    {
        if ( returnOnlyMesh )
            return unexpected( "File contains a point cloud and not a mesh: " + utf8string( file ) );

        if ( edges )
        {
            auto polyline = std::make_shared<Polyline3>();
            polyline->points = std::move( mesh->points );
            polyline->topology.vertResize( polyline->points.size() );
            polyline->topology.makeEdges( *edges );

            auto objectLines = std::make_unique<ObjectLines>();
            objectLines->setName( utf8string( file.stem() ) );
            objectLines->setPolyline( polyline );

            if ( !colors.empty() )
            {
                objectLines->setVertsColorMap( std::move( colors ) );
                objectLines->setColoringType( ColoringType::VertsColorMap );
            }

            objectLines->setXf( xf );

            return LoadedObject{ .obj = std::move( objectLines ) };
        }

        auto pointCloud = std::make_shared<PointCloud>();
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

        return LoadedObject{ .obj = std::move( objectPoints ) };
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

    holesCount = int( objectMesh->numHoles() );
    std::string warnings = makeWarningString( skippedFaceCount, duplicatedVertexCount, holesCount );
        if ( !colors.empty() && !hasColors )
        warnings += fmt::format( "Ignoring too few ({}) colors loaded for a mesh with {} vertices.\n", colors.size(), numVerts );
        if ( !uvCoords.empty() && !hasUV )
        warnings += fmt::format( "Ignoring too few ({}) uv-coordinates loaded for a mesh with {} vertices.\n", uvCoords.size(), numVerts );

    return LoadedObject{ .obj = std::move( objectMesh ), .warnings = std::move( warnings ) };
}

Expected<ObjectLines> makeObjectLinesFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    VertColors colors;
    LinesLoadSettings settings
    {
        .colors = &colors,
        .callback = callback
    };
    auto lines = LinesLoad::fromAnySupportedFormat( file, settings );
    if ( !lines.has_value() )
        return unexpected( std::move( lines.error() ) );

    const auto numVerts = lines->points.size();
    const bool hasColors = colors.size() >= numVerts;

    ObjectLines objectLines;
    objectLines.setName( utf8string( file.stem() ) );
    objectLines.setPolyline( std::make_shared<Polyline3>( std::move( lines.value() ) ) );

    if ( hasColors )
    {
        objectLines.setVertsColorMap( std::move( colors ) );
        objectLines.setColoringType( ColoringType::VertsColorMap );
    }

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
    objectPoints.setPointCloud( std::make_shared<PointCloud>( std::move( pointsCloud.value() ) ) );
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
    auto distanceMap = DistanceMapLoad::fromAnySupportedFormat( file, {
        .distanceMapToWorld = &params,
        .progress = callback,
    } );
    if ( !distanceMap.has_value() )
    {
        return unexpected( distanceMap.error() );
    }

    ObjectDistanceMap objectDistanceMap;
    objectDistanceMap.setName( utf8string( file.stem() ) );
    objectDistanceMap.setDistanceMap( std::make_shared<DistanceMap>( std::move( distanceMap.value() ) ), params );

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

Expected<LoadedObjects> loadObjectFromFile( const std::filesystem::path& filename, const ProgressCallback& callback )
{
    if ( callback && !callback( 0.f ) )
        return unexpectedOperationCanceled();

    Expected<LoadedObjects> result = unexpectedUnsupportedFileExtension();
    bool loadedFromSceneFile = false;

    auto ext = std::string( "*" ) + utf8string( filename.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( findFilter( SceneLoad::getFilters(), ext ) )
    {
        const auto objTree = loadSceneFromAnySupportedFormat( filename, callback );
        if ( !objTree.has_value() )
            return unexpected( std::move( objTree.error() ) );

        objTree->obj->setName( utf8string( filename.stem() ) );
        result = LoadedObjects{ .objs = { objTree->obj }, .warnings = std::move( objTree->warnings ) };
        loadedFromSceneFile = true;
    }
    else if ( const auto filter = findFilter( ObjectLoad::getFilters(), ext ) )
    {
        const auto loader = ObjectLoad::getObjectLoader( *filter );
        result = loader( filename, callback );
    }
    // no else to support same extensions in object and mesh loaders
    if ( !result.has_value() && result.error() != stringOperationCanceled() )
    {
        auto maybe = makeObjectFromMeshFile( filename, callback );
        if ( maybe )
        {
            maybe->obj->select( true );
            result = LoadedObjects{ .objs = { maybe->obj }, .warnings = std::move( std::move( maybe->warnings ) ) };
        }
        else if ( !maybe.error().starts_with( stringUnsupportedFileExtension() ) )
            result = unexpected( std::move( maybe.error() ) );
    }

    if ( !result.has_value() && result.error() != stringOperationCanceled() )
    {
        auto objectLines = makeObjectLinesFromFile( filename, callback );
        if ( objectLines.has_value() )
        {
            objectLines->select( true );
            auto obj = std::make_shared<ObjectLines>( std::move( objectLines.value() ) );
            result = LoadedObjects{ .objs = { obj } };
        }
        else if ( objectLines.error() != stringUnsupportedFileExtension() )
            result = unexpected( std::move( objectLines.error() ) );
    }

    if ( !result.has_value() && result.error() != stringOperationCanceled() )
    {
        auto objectPoints = makeObjectPointsFromFile( filename, callback );
        if ( objectPoints.has_value() )
        {
            objectPoints->select( true );
            auto obj = std::make_shared<ObjectPoints>( std::move( objectPoints.value() ) );
            result = LoadedObjects{ .objs = { obj } };
        }
        else if ( objectPoints.error() != stringUnsupportedFileExtension() )
            result = unexpected( std::move( objectPoints.error() ) );
    }

    if ( !result.has_value() && result.error() != stringOperationCanceled() )
    {
        auto objectDistanceMap = makeObjectDistanceMapFromFile( filename, callback );
        if ( objectDistanceMap.has_value() )
        {
            objectDistanceMap->select( true );
            auto obj = std::make_shared<ObjectDistanceMap>( std::move( objectDistanceMap.value() ) );
            result = LoadedObjects{ .objs = { obj } };
        }
        else if ( objectDistanceMap.error() != stringUnsupportedFileExtension() )
            result = unexpected( std::move( objectDistanceMap.error() ) );
    }

    if ( !result.has_value() && result.error() != stringOperationCanceled() )
    {
        auto objectGcode = makeObjectGcodeFromFile( filename, callback );
        if ( objectGcode.has_value() )
        {
            objectGcode->select( true );
            auto obj = std::make_shared<ObjectGcode>( std::move( objectGcode.value() ) );
            result = LoadedObjects{ .objs = { obj } };
        }
        else if ( objectGcode.error() != stringUnsupportedFileExtension() )
            result = unexpected( std::move( objectGcode.error() ) );
    }

    if ( result.has_value() && !loadedFromSceneFile )
        for ( const std::shared_ptr<Object>& o : result->objs )
        {
            postImportObject( o, filename );
            if ( auto objectPoints = o->asType<ObjectPoints>(); objectPoints )
            {
                if ( !objectPoints->pointCloud()->hasNormals() )
                    result->warnings += "Point cloud " + o->name() + " has no normals.\n";
                if ( objectPoints->getRenderDiscretization() > 1 )
                    result->warnings += "Point cloud " + o->name() + " has too many points in PointCloud:\n"
                    "Visualization is simplified (only part of the points is drawn)\n";
            }
        }

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

Expected<LoadedObject> loadSceneFromAnySupportedFormat( const std::filesystem::path& path, const ProgressCallback& callback )
{
    auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto loader = SceneLoad::getSceneLoader( ext );
    if ( !loader )
        return unexpectedUnsupportedFileExtension();

    return loader( path, callback )
    .and_then( [&] ( LoadedObject&& l ) -> Expected<LoadedObject>
    {
        if ( ext != "*.mru" && ext != "*.zip" )
            postImportObject( l.obj, path );

        return std::move( l );
    } );
}

Expected<LoadedObject> deserializeObjectTree( const std::filesystem::path& path, const FolderCallback& postDecompress,
                                              const ProgressCallback& progressCb )
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

Expected<LoadedObject> deserializeObjectTreeFromFolder( const std::filesystem::path& folder,
                                                        const ProgressCallback& progressCb )
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
    LoadedObject res;
    for (int i = typeTreeSize-1;i>=0;--i)
    {
        const auto& type = root["Type"][unsigned( i )];
        if ( type.isString() )
            res.obj = createObject( type.asString() );
        if ( res.obj )
            break;
    }
    if ( !res.obj )
        return unexpected( "Unknown root object type" );

    int modelNumber{ 0 };
    int modelCounter{ 0 };
    auto cb = progressCb;
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
        cb = [progressCb, &modelCounter, modelNumber] ( float v )
        {
            return progressCb( ( modelCounter + v ) / modelNumber );
        };
    }

    auto resDeser = res.obj->deserializeRecursive( folder, root, cb, &modelCounter );
    if ( !resDeser.has_value() )
    {
        std::string errorStr = resDeser.error();
        if ( errorStr != stringOperationCanceled() )
            errorStr = "Cannot deserialize: " + errorStr;
        return unexpected( errorStr );
    }

    return res;
}

Expected<LoadedObject> deserializeObjectTree( const std::filesystem::path& path, const ProgressCallback& progressCb )
{
    return deserializeObjectTree( path, FolderCallback{}, progressCb );
}

MR_ADD_SCENE_LOADER_WITH_PRIORITY( IOFilter( "MeshInspector scene (.mru)", "*.mru" ), deserializeObjectTree, -1 )

} //namespace MR
