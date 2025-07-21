#include "MRObjectLines.h"
#include "MRObjectFactory.h"
#include "MRPolyline.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRFmt.h"

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

        res.push_back( "edges: " + std::to_string( numUndirectedEdges() ) );
        if( numUndirectedEdges() < polyline_->topology.undirectedEdgeSize() )
            res.back() += " / " + std::to_string( polyline_->topology.undirectedEdgeSize() ) + " size";
        if( polyline_->topology.undirectedEdgeSize() < polyline_->topology.undirectedEdgeCapacity() )
            res.back() += " / " + std::to_string( polyline_->topology.undirectedEdgeCapacity() ) + " capacity";

        res.push_back( fmt::format( "total length: {:.6}", totalLength() ) );
        res.push_back( fmt::format( "avg edge len: {:.6}", avgEdgeLen() ) );

        boundingBoxToInfoLines_( res );
    }
    else
    {
        res.push_back( "no polyline" );
    }

    return res;
}

std::shared_ptr<ObjectLines> merge( const std::vector<std::shared_ptr<ObjectLines>>& objsLines )
{
    MR_TIMER;

    size_t totalVerts = 0;
    bool hasVertColorMap = false; // least one input line has
    for ( const auto& obj : objsLines )
    {
        if ( !obj->polyline() )
            continue;
        totalVerts += obj->polyline()->topology.numValidVerts();
        if ( !obj->getVertsColorMap().empty() )
            hasVertColorMap = true;
    }

    VertColors vertColors;
    if ( hasVertColorMap )
        vertColors.resizeNoInit( totalVerts );

    auto line = std::make_shared<Polyline3>();
    auto& points = line->points;
    points.reserve( totalVerts );
    line->topology.vertReserve( totalVerts );

    for ( const auto& obj : objsLines )
    {
        if ( !obj->polyline() )
            continue;

        VertMap srcToMergeVmap;
        UndirectedEdgeBitSet validPoints;
        validPoints.resize( obj->polyline()->topology.undirectedEdgeSize(), true );
        line->addPartByMask( *obj->polyline(), validPoints, &srcToMergeVmap );

        auto worldXf = obj->worldXf();
        for ( const auto& vInd : srcToMergeVmap )
        {
            if ( vInd.valid() )
                points[vInd] = worldXf( points[vInd] );
        }

        if ( hasVertColorMap )
        {
            const auto& curColorMap = obj->getVertsColorMap();
            for ( VertId thisId = 0_v; thisId < srcToMergeVmap.size(); ++thisId )
            {
                if ( auto mergeId = srcToMergeVmap[thisId] )
                    vertColors[mergeId] = curColorMap.size() <= thisId ? obj->getFrontColor() : curColorMap[thisId];
            }
        }
    }

    assert( points.size() == totalVerts );
    assert( line->topology.vertSize() == totalVerts );

    auto objectLines = std::make_shared<ObjectLines>();
    objectLines->setPolyline( std::move( line ) );
    objectLines->setVertsColorMap( std::move( vertColors ) );
    if( hasVertColorMap )
        objectLines->setColoringType( ColoringType::VertsColorMap );
    return objectLines;
}

std::shared_ptr<ObjectLines> cloneRegion( const std::shared_ptr<ObjectLines>& objLines, const UndirectedEdgeBitSet& region )
{
    MR_TIMER;
    std::shared_ptr<Polyline3> newPolyline = std::make_shared<Polyline3>();
    VertMap src2clone;
    newPolyline->addPartByMask( *objLines->polyline(), region, &src2clone );
    std::shared_ptr<ObjectLines> newObj = std::make_shared<ObjectLines>();
    newObj->setFrontColor( objLines->getFrontColor( true ), true );
    newObj->setFrontColor( objLines->getFrontColor( false ), false );
    newObj->setBackColor( objLines->getBackColor() );
    newObj->setPolyline( newPolyline );
    newObj->setAllVisualizeProperties( objLines->getAllVisualizeProperties() );

    VertMap clone2src;
    clone2src.resizeNoInit( newPolyline->points.size() );
    ParallelFor( src2clone, [&] ( VertId srcV )
    {
        if( auto cloneV = src2clone[srcV] )
            clone2src[cloneV] = srcV;
    } );
    newObj->copyColors( *objLines, clone2src );
    newObj->setName( objLines->name() + "_part" );
    return newObj;
}

} // namespace MR
