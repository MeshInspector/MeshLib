#include "MRViewerIO.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRVoxelsSave.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRDistanceMapSave.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRMeshLoadObj.h"
#include "MRViewerInstance.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRImageLoad.h"

namespace
{

using namespace MR;

tl::expected<MeshLoad::MtlLibrary, std::string> loadMtlLibrary( const std::filesystem::path& filepath )
{
    if ( !std::filesystem::exists( filepath ) )
        return tl::make_unexpected( "File does not exist" );

    std::ifstream in( filepath );
    if ( !in )
        return tl::make_unexpected( "Failed to open file" );

    Buffer<char> buffer( std::filesystem::file_size( filepath ) );
    if ( !in.read( buffer.data(), (ptrdiff_t)buffer.size() ) )
        return tl::make_unexpected( "Failed to read file" );

    return MeshLoad::loadMtlLibrary( buffer.data(), buffer.size() );
}

std::string fixSeparator( std::string str )
{
    std::replace( str.begin(), str.end(), '\\', '/' );
    return str;
}

void applyMaterialLibrary( ObjectMesh& objectMesh, const std::filesystem::path& filename, const MeshLoad::NamedMesh& namedMesh, const MeshLoad::MtlLibrary& materialLibrary )
{
    // TODO: build a single texture
    std::set<std::string> textures;
    for ( const auto& materialVertMap : namedMesh.materialVertMaps )
    {
        auto it = materialLibrary.find( materialVertMap.materialName );
        if ( it == materialLibrary.end() )
            continue;

        auto& material = it->second;
        if ( !material.diffuseTextureFile.empty() )
            textures.insert( material.diffuseTextureFile );
        else if ( material.diffuseColor.lengthSq() != 0 )
            textures.insert( "" );
    }
    if ( textures.size() == 1 && !textures.begin()->empty() )
    {
        const auto textureFile = filename.parent_path() / fixSeparator( *textures.begin() );
        auto image = ImageLoad::fromAnySupportedFormat( textureFile );
        if ( !image.has_value() )
            return;

        MeshTexture texture{ std::move( *image ) };
        objectMesh.setTexture( std::move( texture ) );
        objectMesh.setUVCoords( namedMesh.uvCoords );
        objectMesh.setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
        return;
    }

    Vector<Color, VertId> colorMap( namedMesh.uvCoords.size(), Color::white() );
    for ( const auto& materialVertMap : namedMesh.materialVertMaps )
    {
        auto it = materialLibrary.find( materialVertMap.materialName );
        if ( it == materialLibrary.end() )
        {
            spdlog::warn( "OBJ-file requires material \"{}\", but related MTL-file lacks it", materialVertMap.materialName );
            continue;
        }

        auto& material = it->second;
        if ( !material.diffuseTextureFile.empty() )
        {
            const auto textureFile = filename.parent_path() / fixSeparator( material.diffuseTextureFile );
            auto image = ImageLoad::fromAnySupportedFormat( textureFile );
            if ( !image.has_value() )
                continue;

            BitSetParallelFor( materialVertMap.vertices, [&] ( VertId v )
            {
                auto uv = namedMesh.uvCoords[v];
                uv.x = std::fmod( uv.x, 1.f ) + float( uv.x < 0.f );
                uv.y = std::fmod( uv.y, 1.f ) + float( uv.y < 0.f );
                int x = uv.x * ( image->resolution.x - 1 );
                int y = uv.y * ( image->resolution.y - 1 );
                colorMap[v] = image->pixels[x + y * image->resolution.x];
            } );
        }
        else
        {
            BitSetParallelFor( materialVertMap.vertices, [&] ( VertId v )
            {
                colorMap[v] = Color( material.diffuseColor );
            } );
        }
    }
    objectMesh.setVertsColorMap( std::move( colorMap ) );
    objectMesh.setColoringType( ColoringType::VertsColorMap );
}

}

namespace MR
{

VoidOrErrStr saveObjectToFile( const Object& obj, const std::filesystem::path& filename, ProgressCallback callback )
{
    if ( callback && !callback( 0.f ) )
        return tl::make_unexpected( std::string( "Saving canceled" ) );

    VoidOrErrStr result;

    if ( auto objPoints = obj.asType<ObjectPoints>() )
    {
        if ( objPoints->pointCloud() )
        {
            const auto& colors = objPoints->getVertsColorMap();
            result = PointsSave::toAnySupportedFormat( *objPoints->pointCloud(), filename,
                                                         colors.empty() ? nullptr : &colors, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectPoints has no PointCloud in it" ) );
    }
    else if ( auto objLines = obj.asType<ObjectLines>() )
    {
        if ( objLines->polyline() )
        {
            result = LinesSave::toAnySupportedFormat( *objLines->polyline(), filename, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectLines has no Polyline in it" ) );
    }
    else if ( auto objMesh = obj.asType<ObjectMesh>() )
    {
        if ( objMesh->mesh() )
        {
            const Vector<Color, VertId>* colors{ nullptr };
            if ( objMesh->getColoringType() == ColoringType::VertsColorMap )
                colors = &objMesh->getVertsColorMap();

            result = MeshSave::toAnySupportedFormat( *objMesh->mesh(), filename, colors, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectMesh has no Mesh in it" ) );
    }
    else if ( auto objDistanceMap = obj.asType<ObjectDistanceMap>() )
    {
        if ( auto distanceMap = objDistanceMap->getDistanceMap() )
        {
            result = DistanceMapSave::toAnySupportedFormat( filename, *distanceMap, &objDistanceMap->getToWorldParameters() );
        }
        else
        {
            result = tl::make_unexpected( std::string( "ObjectDistanceMap has no DistanceMap in it" ) );
        }
    }
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
    else if ( auto objVoxels = obj.asType<ObjectVoxels>() )
    {
        auto ext = filename.extension().u8string();
        for ( auto& c : ext )
            c = ( char )tolower( c );

        result = VoxelsSave::toAnySupportedFormat( filename, objVoxels->vdbVolume(), callback );
    }
#endif

    if ( !result.has_value() )
        spdlog::error( result.error() );

    return result;
}

}
