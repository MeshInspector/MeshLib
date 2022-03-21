#include "MRObjectLines.h"
#include "MRObjectFactory.h"
#include "MRPolyline.h"
#include "MRSerializer.h"
#include "MRTimer.h"
#include "MRMeshTexture.h"
#include "MRPch/MRJson.h"
#include "MRSceneColors.h"
#include "MRPch/MRTBB.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectLines )

std::shared_ptr<Object> ObjectLines::clone() const
{
    auto res = std::make_shared<ObjectLines>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = std::make_shared<Polyline3>( *polyline_ );
    return res;
}

std::shared_ptr<Object> ObjectLines::shallowClone() const
{
    auto res = std::make_shared<ObjectLines>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = polyline_;
    return res;
}

void ObjectLines::setPolyline( const std::shared_ptr<Polyline3>& polyline )
{
    polyline_ = polyline;
    setDirtyFlags( DIRTY_ALL );
}

void ObjectLines::setDirtyFlags( uint32_t mask )
{
    VisualObject::setDirtyFlags( mask );

    if ( mask & DIRTY_POSITION || mask & DIRTY_PRIMITIVES )
    {
        totalLength_.reset();
        worldBox_.reset();
        if ( polyline_ )
            polyline_->invalidateCaches();
    }
}

void ObjectLines::swapBase_( Object& other )
{
    if ( auto otherLines = other.asType<ObjectLines>() )
        std::swap( *this, *otherLines );
    else
        assert( false );
}

std::vector<std::string> ObjectLines::getInfoLines() const
{
    std::vector<std::string> res;
    res.push_back( "type : ObjectLines" );

    std::stringstream ss;
    if ( polyline_ )
    {
        ss << "vertices : " << polyline_->topology.numValidVerts();
        res.push_back( ss.str() );

        if ( !totalLength_ )
            totalLength_ = polyline_->totalLength();
        res.push_back( "total length : " + std::to_string( *totalLength_ ) );

        boundingBoxToInfoLines_( res );
    }
    else
    {
        res.push_back( "no polyline" );
    }
    
    return res;
}

ObjectLines::ObjectLines( const ObjectLines& other ) :
    ObjectLinesHolder( other )
{
}

}
