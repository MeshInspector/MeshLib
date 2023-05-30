#include "MRObjectGcode.h"
#include "MRObjectFactory.h"
#include "MRPolyline.h"
#include "MRPch/MRJson.h"
#include "MRSerializer.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectGcode )

ObjectGcode::ObjectGcode()
{
    setVisualizeProperty( true, LinesVisualizePropertyType::Smooth, ViewportMask::all() );
    setColoringType( ColoringType::VertsColorMap );
}



std::shared_ptr<Object> MR::ObjectGcode::clone() const
{
    auto res = std::make_shared<ObjectGcode>( ProtectedStruct{}, *this );
    if ( gcodeSource_ )
        res->gcodeSource_ = std::make_shared<GcodeSource>( *gcodeSource_ );
    return res;
}

std::shared_ptr<Object> ObjectGcode::shallowClone() const
{
    auto res = std::make_shared<ObjectGcode>( ProtectedStruct{}, *this );
    if ( gcodeSource_ )
        res->gcodeSource_ = gcodeSource_;
    return res;
}

void ObjectGcode::setGcodeSource( const std::shared_ptr<GcodeSource>& gcodeSource )
{
    if ( !gcodeSource )
    {
        polyline_ = std::make_shared<Polyline3>();
        setDirtyFlags( DIRTY_ALL );
        return;
    }

    gcodeSource_ = gcodeSource;
    GcodeExecutor executor;
    executor.setGcodeSource( *gcodeSource );
    actionList_ = executor.processSource();

    maxFeedrate_ = 0.f;
    std::shared_ptr<Polyline3> polyline = std::make_shared<Polyline3>();
    for ( int i = 0; i < actionList_.size(); ++i )
    {
        const auto& part = actionList_[i];
        if ( part.path.empty() )
            continue;
        polyline->addFromPoints( part.path.data(), part.path.size() );
        if ( part.feedrate > maxFeedrate_ )
            maxFeedrate_ = part.feedrate;
    }
    polyline_ = polyline;
    updateColors_();
    setDirtyFlags( DIRTY_ALL );
}

void ObjectGcode::setDirtyFlags( uint32_t mask )
{
    ObjectLinesHolder::setDirtyFlags( mask );

    if ( mask & DIRTY_POSITION || mask & DIRTY_PRIMITIVES )
    {
        if ( polyline_ )
        {
            gcodeChangedSignal( mask );
        }
    }
}

std::vector<std::string> ObjectGcode::getInfoLines() const
{
    std::vector<std::string> res = ObjectLinesHolder::getInfoLines();

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

void ObjectGcode::switchFeedrateGradient( bool isFeedrateGradientEnabled )
{
    if ( feedrateGradientEnabled_ == isFeedrateGradientEnabled )
        return;
    feedrateGradientEnabled_ = isFeedrateGradientEnabled;
    updateColors_();
}

void ObjectGcode::setIdleColor( const Color& color )
{
    if ( idleColor_ == color )
        return;
    idleColor_ = color;

    updateColors_();
}

bool ObjectGcode::select( bool isSelected )
{
    if ( !ObjectLinesHolder::select( isSelected ) )
        return false;
    setColoringType( isSelected ? ColoringType::VertsColorMap : ColoringType::SolidColor );
    return true;
}

void ObjectGcode::setFrontColor( const Color& color, bool selected, ViewportId viewportId /*= {} */ )
{
    ObjectLinesHolder::setFrontColor( color, selected, viewportId );
    if ( selected )
        updateColors_();
}

ObjectGcode::ObjectGcode( const ObjectGcode& other ) :
    ObjectLinesHolder( other )
{
}

void ObjectGcode::swapBase_( Object& other )
{
    if ( auto otherGcode = other.asType<ObjectGcode>() )
        std::swap( *this, *otherGcode );
    else
        assert( false );
}

void ObjectGcode::swapSignals_( Object& other )
{
    ObjectLinesHolder::swapSignals_( other );
    if ( auto otherGcode = other.asType<ObjectGcode>() )
        std::swap( gcodeChangedSignal, otherGcode->gcodeChangedSignal );
    else
        assert( false );
}

void ObjectGcode::serializeFields_( Json::Value& root ) const
{
    ObjectLinesHolder::serializeFields_( root );
    root["Type"].append( ObjectGcode::TypeName() );
    root["FeedrateGradientEnable"] = feedrateGradientEnabled_;
    root["MaxFeedrate"] = maxFeedrate_;
    serializeToJson( idleColor_, root["IdleColor"] );

    auto& gcodeSource = root["GcodeSource"];
    for ( const auto& str : *gcodeSource_ )
    {
        Json::Value val;
        val = str;
        gcodeSource.append( val );
    }
}

void ObjectGcode::deserializeFields_( const Json::Value& root )
{
    ObjectLinesHolder::deserializeFields_( root );
    deserializeFromJson( root["IdleColor"], idleColor_ );

    if ( root["FeedrateGradientEnable"].isBool() )
        feedrateGradientEnabled_ = root["FeedrateGradientEnable"].asBool();
    if ( root["MaxFeedrate"].isDouble() )
        maxFeedrate_ = float( root["MaxFeedrate"].asDouble() );

    const auto& gcodeSourceRoot = root["GcodeSource"];
    if ( !gcodeSourceRoot.isArray() )
        return;

    GcodeSource gcodeSource( gcodeSourceRoot.size() );
    for ( int i = 0; i < gcodeSource.size(); ++i )
    {
        if ( gcodeSourceRoot[i].isString() )
            gcodeSource[i] = gcodeSourceRoot[i].asString();
    }

    GcodeExecutor executor;
    executor.setGcodeSource( gcodeSource );
    actionList_ = executor.processSource();

    updateColors_();
}

void ObjectGcode::updateColors_()
{
    const bool feedrateValid = maxFeedrate_ > 0.f;
    VertColors colors;
    const Color workColor = getFrontColor( true );
    for ( int i = 0; i < actionList_.size(); ++i )
    {
        const auto& part = actionList_[i];
        if ( part.path.empty() )
            continue;
        Color color = idleColor_;
        if ( !part.idle )
        {
            if ( feedrateGradientEnabled_ && feedrateValid )
            {
                color = workColor * ( 0.3f + 0.7f * part.feedrate / maxFeedrate_ );
                color.a = 255;
            }
            else
                color = workColor;
        }
        colors.autoResizeSet( VertId( colors.size() ), part.path.size(), color );
    }
    setVertsColorMap( colors );
}

}
