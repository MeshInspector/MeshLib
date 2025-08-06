#include "MRObjectDistanceMap.h"
#include "MRDirectory.h"
#include "MRObjectFactory.h"
#include "MRDistanceMap.h"
#include "MRSerializer.h"
#include "MRDistanceMapSave.h"
#include "MRDistanceMapLoad.h"
#include "MRSceneColors.h"
#include "MRMesh.h"
#include "MRHeapBytes.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MRStringConvert.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectDistanceMap )

void ObjectDistanceMap::applyScale( float scaleFactor )
{
    dmap2local_.A *= scaleFactor;
    dmap2local_.b *= scaleFactor;

    ObjectMeshHolder::applyScale( scaleFactor );
}

std::shared_ptr<MR::Object> ObjectDistanceMap::clone() const
{
    auto res = std::make_shared<ObjectDistanceMap>( ProtectedStruct{}, *this );
    if ( data_.mesh )
        res->data_.mesh = std::make_shared<Mesh>( *data_.mesh );
    if ( dmap_ )
        res->dmap_ = std::make_shared<DistanceMap>( *dmap_ );
    return res;
}

std::shared_ptr<MR::Object> ObjectDistanceMap::shallowClone() const
{
    return std::make_shared<ObjectDistanceMap>( ProtectedStruct{}, *this );
}

void ObjectDistanceMap::swapBase_( Object& other )
{
    if ( auto otherObject = other.asType<ObjectDistanceMap>() )
        std::swap( *this, *otherObject );
    else
        assert( false );
}

std::vector<std::string> ObjectDistanceMap::getInfoLines() const
{
    std::vector<std::string> res = ObjectMeshHolder::getInfoLines();

    std::stringstream ss;
    if ( dmap_ )
    {
        ss << "DMap resolution:"
            << "\n resX = " << dmap_->resX()
            << "\n resY = " << dmap_->resY();
        boundingBoxToInfoLines_( res );
    }
    else
    {
        ss << "no distance map";
    }
    res.push_back( ss.str() );
    ss.str( "" );

    ss << std::setprecision( 4 );
    auto printVector3 = [&ss] ( const char* name, const Vector3f& v )
    {
        ss << "\n " << name << ":"
            << "\n  x = " << v.x
            << "\n  y = " << v.y
            << "\n  z = " << v.z;
    };
    ss << "Params:";
    printVector3( "pixelXVec", dmap2local_.A.col( 0 ) );
    printVector3( "pixelYVec", dmap2local_.A.col( 1 ) );
    printVector3( "depthVec", dmap2local_.A.col( 2 ) );
    printVector3( "origin", dmap2local_.b );
    res.push_back( ss.str() );

    return res;
}

bool ObjectDistanceMap::setDistanceMap( const std::shared_ptr<DistanceMap>& dmap, const AffineXf3f& map2local, bool updateMesh, ProgressCallback cb )
{
    return construct_( dmap, map2local, updateMesh, cb );
}

std::shared_ptr<Mesh> ObjectDistanceMap::calculateMesh( ProgressCallback cb ) const
{
    auto res = distanceMapToMesh( *dmap_, dmap2local_, cb );
    if ( !res.has_value() )
    {
        return nullptr;
    }
    return std::make_shared<Mesh>( res.value() );
}

void ObjectDistanceMap::updateMesh( const std::shared_ptr<Mesh>& mesh )
{
    data_.mesh = mesh;
    setDirtyFlags( DIRTY_ALL );
}

size_t ObjectDistanceMap::heapBytes() const
{
    return ObjectMeshHolder::heapBytes() + MR::heapBytes( dmap_ );
}

ObjectDistanceMap::ObjectDistanceMap()
{
    setDefaultSceneProperties_();
}

void ObjectDistanceMap::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    serializeToJson( dmap2local_.A.col( 0 ), root["PixelXVec"] );
    serializeToJson( dmap2local_.A.col( 1 ), root["PixelYVec"] );
    serializeToJson( dmap2local_.A.col( 2 ), root["DepthVec"] );
    serializeToJson( dmap2local_.b, root["OriginWorld"] );

    root["Type"].append( ObjectDistanceMap::TypeName() );
}

void ObjectDistanceMap::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    Vector3f orgPoint;
    Vector3f pixelXVec{ Vector3f::plusX() };
    Vector3f pixelYVec{ Vector3f::plusY() };
    Vector3f direction{ Vector3f::plusZ() };
    deserializeFromJson( root["PixelXVec"], pixelXVec );
    deserializeFromJson( root["PixelYVec"], pixelYVec );
    deserializeFromJson( root["DepthVec"], direction );
    deserializeFromJson( root["OriginWorld"], orgPoint );
    dmap2local_ = { Matrix3f::fromColumns( pixelXVec, pixelYVec, direction ), orgPoint };

    if ( root["UseDefaultSceneProperties"].isBool() && root["UseDefaultSceneProperties"].asBool() )
        setDefaultSceneProperties_();

    construct_( dmap_, dmap2local_ );
}

Expected<void> ObjectDistanceMap::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    auto modelPath = pathFromUtf8( utf8string( path ) + saveDistanceMapFormat_ );
    std::error_code ec;
    if ( !std::filesystem::is_regular_file( modelPath, ec ) )
    {
        modelPath = findPathWithExtension( path );
        if ( modelPath.empty() )
            return unexpected( "No distance map file found: " + utf8string( path ) );
    }

    auto res = DistanceMapLoad::fromAnySupportedFormat( modelPath, { .progress = progressCb } );
    if ( !res.has_value() )
        return unexpected( res.error() );
    
    dmap_ = std::make_shared<DistanceMap>( res.value() );
    return {};
}

Expected<std::future<Expected<void>>> ObjectDistanceMap::serializeModel_( const std::filesystem::path& path ) const
{
    if ( !dmap_ )
        return {};

    return std::async( getAsyncLaunchType(), [this, filename = utf8string( path ) + saveDistanceMapFormat_] ()
    {
        return DistanceMapSave::toAnySupportedFormat( *dmap_, pathFromUtf8( filename ) );
    } );
}

void ObjectDistanceMap::resetFrontColor()
{
    // cannot implement in the opposite way to keep `setDefaultColors_()` non-virtual
    setDefaultColors_();
}

void ObjectDistanceMap::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectDistanceMap ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectDistanceMap ), false );
}

bool ObjectDistanceMap::construct_( const std::shared_ptr<DistanceMap>& dmap, const AffineXf3f& dmap2local, bool needUpdateMesh, ProgressCallback cb )
{
    if ( !dmap )
        return false;

    dmap_ = dmap;
    dmap2local_ = dmap2local;

    if ( needUpdateMesh )
    {
        auto mesh = calculateMesh( cb );
        if ( !mesh )
        {
            return false;
        }

        updateMesh( mesh );
    }


    return true;
}

void ObjectDistanceMap::setDefaultSceneProperties_()
{
    setDefaultColors_();
}

} // namespace MR
