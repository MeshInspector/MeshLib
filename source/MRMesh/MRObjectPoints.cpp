#include "MRObjectPoints.h"
#include "MRObjectFactory.h"
#include "MRMeshToPointCloud.h"
#include "MRObjectMesh.h"
#include "MRRegionBoundary.h"
#include "MRMesh.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"

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

std::vector<std::string> ObjectPoints::getInfoLines() const
{
    std::vector<std::string> res = ObjectPointsHolder::getInfoLines();

    if ( points_ )
    {
        if ( points_->normals.empty() )
            res.push_back( "points: " );
        else
            res.push_back( "points with normals: " );
        const auto nValidPoints = numValidPoints();
        res.back() += std::to_string( nValidPoints );

        const auto nSelectedPoints = numSelectedPoints();
        if( nSelectedPoints )
            res.back() += " / " + std::to_string( nSelectedPoints ) + " selected";

        if( nValidPoints < points_->points.size() )
            res.back() += " / " + std::to_string( points_->points.size() ) + " size";

        if( points_->points.size() < points_->points.capacity() )
            res.back() += " / " + std::to_string( points_->points.capacity() ) + " capacity";

        boundingBoxToInfoLines_( res );
    }
    else
        res.push_back( "no points" );

    return res;
}

void ObjectPoints::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    ObjectPointsHolder::setDirtyFlags( mask, invalidateCaches );
    if ( points_ )
    {
        if ( mask & DIRTY_POSITION || mask & DIRTY_FACE )
            pointsChangedSignal( mask );
        if ( mask & DIRTY_RENDER_NORMALS )
            normalsChangedSignal( mask );
    }
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

void ObjectPoints::swapPointCloud( std::shared_ptr< PointCloud >& points )
{
    if ( points == points_ )
        return;
    points_.swap( points );
    setDirtyFlags( DIRTY_ALL );
}

void ObjectPoints::swapBase_( Object& other )
{
    if ( auto otherPoints = other.asType<ObjectPoints>() )
        std::swap( *this, *otherPoints );
    else
        assert( false );
}

void ObjectPoints::swapSignals_( Object& other )
{
    ObjectPointsHolder::swapSignals_( other );
    if ( auto otherPoints = other.asType<ObjectPoints>() )
    {
        std::swap( pointsChangedSignal, otherPoints->pointsChangedSignal );
        std::swap( normalsChangedSignal, otherPoints->normalsChangedSignal );
    }
    else
        assert( false );
}

void ObjectPoints::serializeFields_( Json::Value& root ) const
{
    ObjectPointsHolder::serializeFields_( root );

    root["Type"].append( ObjectPoints::TypeName() );
}

}
