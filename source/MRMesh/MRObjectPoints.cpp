#include "MRObjectPoints.h"
#include "MRObjectFactory.h"
#include "MRPlane3.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshToPointCloud.h"
#include "MRObjectMesh.h"
#include "MRRegionBoundary.h"
#include "MRMesh.h"
#include "MRPointsSave.h"
#include "MRPointsLoad.h"
#include "MRTimer.h"
#include "MRPch/MRJson.h"
#include "MRSceneColors.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRAsyncLaunchType.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectPoints )

ObjectPoints::ObjectPoints( const ObjectMesh& objMesh, bool saveNormals/*=true*/ )
{
    if ( !objMesh.mesh() )
        return;

    const auto verts = getInnerVerts( objMesh.mesh()->topology, objMesh.getSelectedFaces() );
    setPointCloud( std::make_shared<PointCloud>( meshToPointCloud( *objMesh.mesh(), saveNormals, verts.count() > 0 ? &verts : nullptr) ) );
    setName( objMesh.name() + " Points" );
    setVertsColorMap( objMesh.getVertsColorMap() );
    setFrontColor( objMesh.getFrontColor( true ), true );
    setFrontColor( objMesh.getFrontColor( false ), false );
    setBackColor( objMesh.getBackColor() );
    setColoringType( objMesh.getColoringType() );
}

ObjectPoints::ObjectPoints()
{
    setDefaultColors_();
}

void ObjectPoints::applyScale( float scaleFactor )
{
    if ( !points_ )
        return;

    auto& points = points_->points;

    tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )points.size() ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            points[VertId( i )] *= scaleFactor;
        }
    } );
    setDirtyFlags( DIRTY_POSITION );
}

Box3f ObjectPoints::getWorldBox() const
{
    if ( !points_ )
        return {};
    const auto worldXf = this->worldXf();
    if ( auto v = worldBox_.get( worldXf ) )
        return *v;
    const auto box = points_->computeBoundingBox( &worldXf );
    worldBox_.set( worldXf, box );
    return box;
}

std::vector<std::string> ObjectPoints::getInfoLines() const
{
    std::vector<std::string> res;
    res.push_back( "type : ObjectPoints" );

    if ( points_ )
    {
        std::stringstream ss;
        ss << "PointCloud :"
            << "\n Size : " << points_->points.size();
        res.push_back( ss.str() );
        boundingBoxToInfoLines_( res );
    }
    else
        res.push_back( "no points" );

    return res;
}

std::shared_ptr<Object> ObjectPoints::clone() const
{
    auto res = std::make_shared<ObjectPoints>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = std::make_shared<PointCloud>( *points_ );
    return res;
}

std::shared_ptr<Object> ObjectPoints::shallowClone() const
{
    auto res = std::make_shared<ObjectPoints>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = points_;
    return res;
}

void ObjectPoints::setDirtyFlags( uint32_t mask )
{
    VisualObject::setDirtyFlags( mask );

    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE )
    {
        worldBox_.reset();
        if ( points_ )
            points_->invalidateCaches();
    }
}

void ObjectPoints::setPointSize( float size )
{
    if ( pointSize_ == size )
        return;

    pointSize_ = size;
    needRedraw_ = true;
}

void ObjectPoints::swapBase_( Object& other )
{
    if ( auto otherPoints = other.asType<ObjectPoints>() )
        std::swap( *this, *otherPoints );
    else
        assert( false );
}

Box3f ObjectPoints::computeBoundingBox_() const
{
    if( !points_ )
        return {};
    tbb::enumerable_thread_specific<Box3f> threadData;
    BitSetParallelFor( points_->validPoints, [&]( VertId id )
        {
            threadData.local().include( points_->points[id] );
        } );
    Box3f bb;
    for( const auto& b : threadData )
        bb.include( b );
    return bb;
}

Box3f ObjectPoints::computeBoundingBoxXf_() const
{
    if( !points_ )
        return {};
    tbb::enumerable_thread_specific<Box3f> threadData;
    auto wXf = worldXf();
    BitSetParallelFor( points_->validPoints, [&]( VertId id )
        {
            threadData.local().include( wXf (points_->points[id]) );
        } );
    Box3f bb;
    for( const auto& b : threadData )
        bb.include( b );
    return bb;
}

tl::expected<std::future<void>, std::string> ObjectPoints::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !points_ )
        return {};

    const std::vector<Color>* colorMapPtr = vertsColorMap_.empty() ? nullptr : &vertsColorMap_.vec_;
    return std::async( getAsyncLaunchType(),
        [points = points_, filename = path.u8string() + u8".ctm", ptr = colorMapPtr]() { MR::PointsSave::toCtm( *points, filename, ptr ); } );
}

tl::expected<void, std::string> ObjectPoints::deserializeModel_( const std::filesystem::path& path )
{
    auto res = PointsLoad::fromCtm( path.u8string() + u8".ctm", &vertsColorMap_.vec_ );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );

    if ( !vertsColorMap_.empty() )
        setColoringType( ColoringType::VertsColorMap );

    points_ = std::make_shared<PointCloud>( std::move( res.value() ) );
    return {};
}

void ObjectPoints::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["Type"].append( ObjectPoints::TypeName() );
}

void ObjectPoints::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectPoints>( *this );
}

void ObjectPoints::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectPoints ) );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectPoints ), false );
}

}
