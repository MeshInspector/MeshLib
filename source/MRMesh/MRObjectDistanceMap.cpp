#include "MRObjectDistanceMap.h"
#include "MRObjectFactory.h"
#include "MRDistanceMap.h"
#include "MRSerializer.h"
#include "MRDistanceMapSave.h"
#include "MRDistanceMapLoad.h"
#include "MRSceneColors.h"
#include "MRHeapBytes.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MRStringConvert.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectDistanceMap )

void ObjectDistanceMap::applyScale( float scaleFactor )
{
    toWorldParams_.orgPoint *= scaleFactor;
    toWorldParams_.pixelXVec *= scaleFactor;
    toWorldParams_.pixelYVec *= scaleFactor;
    
    if ( dmap_ )
    {
        tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )(dmap_->resX() * dmap_->resY() ) ),
            [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                if ( dmap_->isValid( i ) )
                    dmap_->getValue( i ) *= scaleFactor;
            }
        } );
    }

    ObjectMeshHolder::applyScale( scaleFactor );
}

std::shared_ptr<MR::Object> ObjectDistanceMap::clone() const
{
    auto res = std::make_shared<ObjectDistanceMap>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    if ( dmap_ )
        res->dmap_ = std::make_shared<DistanceMap>( *dmap_ );
    return res;
}

std::shared_ptr<MR::Object> ObjectDistanceMap::shallowClone() const
{
    auto res = std::make_shared<ObjectDistanceMap>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    if ( dmap_ )
        res->dmap_ = dmap_;
    return res;
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
    printVector3( "pixelXVec", toWorldParams_.pixelXVec );
    printVector3( "pixelYVec", toWorldParams_.pixelYVec );
    printVector3( "depthVec", toWorldParams_.direction );
    printVector3( "origin", toWorldParams_.orgPoint );
    res.push_back( ss.str() );

    return res;
}

void ObjectDistanceMap::setDistanceMap( const std::shared_ptr<DistanceMap>& dmap, const DistanceMapToWorld& params )
{
    dmap_ = dmap;
    toWorldParams_ = params;

    construct_();
}

const std::shared_ptr<MR::DistanceMap>& ObjectDistanceMap::getDistanceMap() const
{
    return dmap_;
}

const DistanceMapToWorld& ObjectDistanceMap::getToWorldParameters() const
{
    return toWorldParams_;
}

size_t ObjectDistanceMap::heapBytes() const
{
    return ObjectMeshHolder::heapBytes() + MR::heapBytes( dmap_ );
}

ObjectDistanceMap::ObjectDistanceMap( const ObjectDistanceMap& other ) :
    ObjectMeshHolder( other ),
    dmap_( nullptr ),
    toWorldParams_( other.toWorldParams_ )
{
}

ObjectDistanceMap::ObjectDistanceMap()
{
    setDefaultColors_();
}

void ObjectDistanceMap::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    serializeToJson( toWorldParams_.pixelXVec, root["PixelXVec"] );
    serializeToJson( toWorldParams_.pixelYVec, root["PixelYVec"] );
    serializeToJson( toWorldParams_.direction, root["DepthVec"] );
    serializeToJson( toWorldParams_.orgPoint, root["OriginWorld"] );

    root["Type"].append( ObjectDistanceMap::TypeName() );
}

void ObjectDistanceMap::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    deserializeFromJson( root["PixelXVec"], toWorldParams_.pixelXVec );
    deserializeFromJson( root["PixelYVec"], toWorldParams_.pixelYVec );
    deserializeFromJson( root["DepthVec"], toWorldParams_.direction );
    deserializeFromJson( root["OriginWorld"], toWorldParams_.orgPoint );

    construct_();
}

tl::expected<void, std::string> ObjectDistanceMap::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    auto res = DistanceMapLoad::loadRaw( utf8string( path ) + ".raw" );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );
    
    dmap_ = std::make_shared<DistanceMap>( res.value().first );
    toWorldParams_ = res.value().second;
    return {};
}

tl::expected<std::future<void>, std::string> ObjectDistanceMap::serializeModel_( const std::filesystem::path& path ) const
{
    if ( !dmap_ )
        return {};

    return std::async( getAsyncLaunchType(),
        [this, filename = utf8string( path ) + ".raw"]() { DistanceMapSave::saveRAW( filename, *dmap_, toWorldParams_ ); } );
}

void ObjectDistanceMap::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectDistanceMap ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectDistanceMap ), false );
}

void ObjectDistanceMap::construct_()
{
    if ( !dmap_ )
        return;

    mesh_ = std::make_shared<Mesh>( distanceMapToMesh( *dmap_, toWorldParams_ ) );
    setDirtyFlags( DIRTY_ALL );
}

} // namespace MR
