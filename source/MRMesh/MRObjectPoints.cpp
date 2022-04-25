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

ObjectPoints::ObjectPoints( const ObjectPoints& other ) :
    ObjectPointsHolder( other )
{
}

std::vector<std::string> ObjectPoints::getInfoLines() const
{
    std::vector<std::string> res;
    res.push_back( "type : ObjectPoints" );

    if ( points_ )
    {
        std::stringstream ss;
        bool showNormalsNum = false;
        if ( points_->normals.empty() )
            res.push_back( "points: " + std::to_string( points_->points.size() ) );
        else if ( points_->points.size() == points_->normals.size() )
            res.push_back( "points with normals: " + std::to_string( points_->points.size() ) );
        else
        {
            res.push_back( "points: " + std::to_string( points_->points.size() ) );
            showNormalsNum = true;
        }

        if ( auto nSelectedVertices = numSelectedVertices() )
            res.back() += " / " + std::to_string( nSelectedVertices ) + " selected";

        if ( showNormalsNum )
            res.push_back( "normals: " + std::to_string( points_->normals.size() ) );

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

void ObjectPoints::swapBase_( Object& other )
{
    if ( auto otherPoints = other.asType<ObjectPoints>() )
        std::swap( *this, *otherPoints );
    else
        assert( false );
}

void ObjectPoints::serializeFields_( Json::Value& root ) const
{
    ObjectPointsHolder::serializeFields_( root );

    root["Type"].append( ObjectPoints::TypeName() );
}

}
