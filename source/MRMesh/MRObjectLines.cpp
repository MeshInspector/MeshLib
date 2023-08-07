#include "MRObjectLines.h"
#include "MRObjectFactory.h"
#include "MRPolyline.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"

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
    if ( polyline_ == polyline )
        return;
    polyline_ = polyline;
    setDirtyFlags( DIRTY_ALL );
}

std::shared_ptr< Polyline3 > ObjectLines::updatePolyline( std::shared_ptr< Polyline3 > polyline )
{
    if ( polyline != polyline_ )
    {
        polyline_.swap( polyline );
        setDirtyFlags( DIRTY_ALL );
    }
    return polyline;
}

void ObjectLines::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    ObjectLinesHolder::setDirtyFlags( mask, invalidateCaches );

    if ( mask & DIRTY_POSITION || mask & DIRTY_PRIMITIVES )
    {
        if ( polyline_ )
        {
            linesChangedSignal( mask );
        }
    }
}

void ObjectLines::swapBase_( Object& other )
{
    if ( auto otherLines = other.asType<ObjectLines>() )
        std::swap( *this, *otherLines );
    else
        assert( false );
}

void ObjectLines::swapSignals_( Object& other )
{
    ObjectLinesHolder::swapSignals_( other );
    if ( auto otherLines = other.asType<ObjectLines>() )
        std::swap( linesChangedSignal, otherLines->linesChangedSignal );
    else
        assert( false );
}

void ObjectLines::serializeFields_( Json::Value& root ) const
{
    ObjectLinesHolder::serializeFields_( root );
    root["Type"].append( ObjectLines::TypeName() );
}

std::vector<std::string> ObjectLines::getInfoLines() const
{
    std::vector<std::string> res = ObjectLinesHolder::getInfoLines();

    if ( polyline_ )
    {
        res.push_back( "components: " + std::to_string( numComponents() ) );

        res.push_back( "vertices: " + std::to_string( polyline_->topology.numValidVerts() ) );
        if( polyline_->topology.numValidVerts() < polyline_->topology.vertSize() )
            res.back() += " / " + std::to_string( polyline_->topology.vertSize() ) + " size";
        if( polyline_->topology.vertSize() < polyline_->topology.vertCapacity() )
            res.back() += " / " + std::to_string( polyline_->topology.vertCapacity() ) + " capacity";

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

}
